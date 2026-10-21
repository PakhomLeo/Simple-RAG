import hmac
import ipaddress
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Sequence, Tuple  # 导入 List

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # 用于处理跨域请求
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend import crud, services  # 使用相对导入
from backend.config import (
    CORS_ALLOW_ORIGINS,
    MAX_UPLOAD_FILE_BYTES,
    MAX_UPLOAD_FILES,
    MILVUS_SEARCH_MAX_DISTANCE,
    SERVE_FRONTEND,
    SIMPLE_RAG_API_KEY,
)
from backend.database import SessionLocal, create_db_and_tables
from backend.models import (
    DocumentUploadResponse,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
    SimilarityScore,
    SourceChunk,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The SQLite table and the Milvus connection are set up in the app lifespan (see `lifespan` below), not at
# import time. `python -m backend.main` with reload on imports this module in the reloader's parent process
# too, and a Milvus Lite file can only be opened by one process.

# Environment variable that `python -m backend.main` sets to the address it binds,
# so the startup check below also works in the reloader's child process.
BIND_HOST_ENV = "SIMPLE_RAG_BIND_HOST"


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _is_uvicorn_cli(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    first = argv[0].replace("\\", "/")
    name = first.rsplit("/", 1)[-1].lower()
    return name in {"uvicorn", "uvicorn.exe"} or first.endswith("uvicorn/__main__.py")


def detect_bind_host(argv: Optional[Sequence[str]] = None, environ: Optional[dict] = None) -> Tuple[Optional[str], str]:
    """Best-effort guess of the address the server listens on.

    Returns ``(host, source)``. ``host`` is ``None`` when the address cannot be determined
    (for example under gunicorn or a custom ``uvicorn.run(...)`` call), and ``"unix-socket"``
    for ``uvicorn --uds``, which is not reachable over the network.
    """
    argv = list(sys.argv if argv is None else argv)
    environ = os.environ if environ is None else environ

    explicit = environ.get(BIND_HOST_ENV, "").strip()
    if explicit:
        return explicit, BIND_HOST_ENV

    if _is_uvicorn_cli(argv):
        host: Optional[str] = None
        uds = False
        fd = False
        args = argv[1:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--host" and i + 1 < len(args):
                host = args[i + 1]
                i += 1
            elif arg.startswith("--host="):
                host = arg.split("=", 1)[1]
            elif arg == "--uds" or arg.startswith("--uds="):
                uds = True
            elif arg == "--fd" or arg.startswith("--fd="):
                fd = True
            i += 1
        if uds:
            return "unix-socket", "uvicorn --uds"
        if fd:
            return None, "uvicorn --fd"
        if host is not None:
            return host, "uvicorn --host"
        env_host = environ.get("UVICORN_HOST", "").strip()
        if env_host:
            return env_host, "UVICORN_HOST"
        return "127.0.0.1", "uvicorn default"

    return None, "unknown launcher"


def enforce_bind_policy(api_key: Optional[str] = None, argv=None, environ=None) -> None:
    """Refuse to start on a non-loopback address without SIMPLE_RAG_API_KEY.

    Runs in the app lifespan, so it covers `uvicorn backend.main:app ...` as well as
    `python -m backend.main`. When the address cannot be determined, log a prominent warning.
    """
    key = SIMPLE_RAG_API_KEY if api_key is None else api_key
    host, source = detect_bind_host(argv, environ)
    if host is None:
        if not key:
            logger.warning(
                "!!! SECURITY WARNING: could not determine the listen address (%s) and SIMPLE_RAG_API_KEY is "
                "not set. Anyone who can reach this server can upload documents and wipe the index. "
                "Bind to 127.0.0.1 or set SIMPLE_RAG_API_KEY. !!!",
                source,
            )
        return
    if host == "unix-socket" or is_loopback_host(host):
        return
    if not key:
        raise RuntimeError(
            f"Refusing to start: listening on non-loopback address {host!r} (from {source}) without "
            "SIMPLE_RAG_API_KEY. Set SIMPLE_RAG_API_KEY or bind to 127.0.0.1."
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    enforce_bind_policy()
    create_db_and_tables()
    services.initialize_services()  # connect to Milvus, load or create the collection, check Ollama
    yield


app = FastAPI(title="Local RAG API", lifespan=lifespan)

# CORS: only the origins listed in CORS_ALLOW_ORIGINS may call the API from a browser.
# The bundled UI is served by this app (same origin) and needs no CORS entry.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """When SIMPLE_RAG_API_KEY is set, require a matching X-API-Key header."""
    if not SIMPLE_RAG_API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), SIMPLE_RAG_API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _safe_display_name(filename: Optional[str]) -> str:
    """Keep only the base name of an uploaded file. The file is never written to disk."""
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    return name[:255] or "unnamed.md"


# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post(
    "/upload/",
    response_model=List[DocumentUploadResponse],
    responses={500: {"model": ErrorResponse}},
    dependencies=[Depends(require_api_key)],
)
async def upload_document(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """
    上传一个或多个 Markdown 文档，处理并存储。
    """
    processed_files_responses = []  # 用于存储每个文件的处理结果

    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f"Too many files. At most {MAX_UPLOAD_FILES} files per upload.")

    # Every upload replaces the whole index: clear Milvus first, then SQLite, so both stores stay in sync.
    # If Milvus cannot be cleared, nothing is deleted from SQLite and the upload is aborted.
    try:
        logger.info("Clearing existing vectors from Milvus before new upload batch...")
        services.clear_milvus_data()
        logger.info("Milvus data cleared successfully (or collection was already empty).")
    except Exception as e:
        logger.error(f"Error clearing Milvus data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear the vector store before upload.")
    try:
        deleted = crud.delete_all_chunks(db)
        logger.info(f"Deleted {deleted} old chunks from SQLite.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error clearing SQLite chunks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear the chunk store before upload.")

    for file in files:  # 遍历文件列表
        display_name = _safe_display_name(file.filename)
        if not display_name.lower().endswith(".md"):
            logger.warning(f"Skipping non-markdown file: {display_name!r}")
            # 可以选择为跳过的文件添加一个特定的响应
            processed_files_responses.append(
                DocumentUploadResponse(
                    filename=display_name,
                    message="Skipped: Invalid file type. Only .md files are allowed.",
                    total_chunks=0,
                )
            )
            continue  # 继续处理下一个文件

        try:
            logger.info(f"Receiving and processing file: {display_name!r}")
            contents = await file.read(MAX_UPLOAD_FILE_BYTES + 1)
            if len(contents) > MAX_UPLOAD_FILE_BYTES:
                processed_files_responses.append(
                    DocumentUploadResponse(
                        filename=display_name,
                        message=f"Skipped: file is larger than {MAX_UPLOAD_FILE_BYTES} bytes.",
                        total_chunks=0,
                    )
                )
                continue
            text_content = contents.decode("utf-8")

            # 1. 解析 Markdown (如果需要复杂解析，否则直接用text_content)
            # plain_text = services.parse_markdown_file(text_content) # 如果上面已是纯文本，则不需要
            # 假设上传的是纯文本的md，或者简单处理，直接用 text_content
            # 如果需要从markdown中提取特定结构（如标题），则需要更复杂的解析
            plain_text = services.parse_markdown_file(text_content)  # 使用 services 中的解析函数
            logger.info(f"Successfully parsed markdown file: {display_name}, length: {len(plain_text)}")

            # 2. 文本切分
            chunks = services.split_text_into_chunks(plain_text)
            if not chunks:
                logger.warning(f"No text chunks generated for file: {display_name}")
                processed_files_responses.append(
                    DocumentUploadResponse(
                        filename=display_name,
                        message="File content is empty or too short to be chunked.",
                        total_chunks=0,
                    )
                )
                continue
            logger.info(f"Split document into {len(chunks)} chunks.")

            # 3. 为每个块生成唯一ID，并存储元数据 (SQLite)
            chunk_ids_in_db = []
            chunk_contents_for_embedding = []
            for chunk_text in chunks:
                db_chunk = crud.create_chunk(db, chunk_content=chunk_text, document_name=display_name)
                chunk_ids_in_db.append(db_chunk.chunk_id)
                chunk_contents_for_embedding.append(chunk_text)
            logger.info(f"Stored {len(chunk_ids_in_db)} chunks metadata in SQLite.")

            # 4. 获取 Embedding
            logger.info(f"Requesting embeddings for {len(chunk_contents_for_embedding)} chunks.")
            embeddings = services.get_embeddings(chunk_contents_for_embedding)
            if len(embeddings) != len(chunk_ids_in_db):
                logger.error("Mismatch between number of embeddings and stored chunks.")
                processed_files_responses.append(
                    DocumentUploadResponse(
                        filename=display_name, message="Embedding generation failed or count mismatch.", total_chunks=0
                    )
                )
                continue
            logger.info(f"Successfully received {len(embeddings)} embeddings.")

            # 5. 存入 Milvus
            services.insert_into_milvus(chunk_ids_in_db, embeddings)
            logger.info(f"Successfully inserted embeddings into Milvus for file: {display_name}")

            processed_files_responses.append(
                DocumentUploadResponse(
                    filename=display_name,
                    message="Document processed and stored successfully.",
                    total_chunks=len(chunks),
                )
            )
        except ValueError as ve:
            logger.error(f"ValueError during processing file {display_name}: {ve}", exc_info=True)
            processed_files_responses.append(
                DocumentUploadResponse(
                    filename=display_name, message=f"Error processing document (ValueError): {str(ve)}", total_chunks=0
                )
            )
        except ConnectionError as ce:  # 更具体的异常捕获
            logger.error(f"ConnectionError during processing file {display_name}: {ce}", exc_info=True)
            processed_files_responses.append(
                DocumentUploadResponse(
                    filename=display_name,
                    message=f"Service connection error during processing: {str(ce)}",
                    total_chunks=0,
                )
            )
            # 如果一个文件的ConnectionError很严重，可能需要停止整个批处理
            # raise HTTPException(status_code=503, detail=f"Service connection error processing {display_name}: {str(ce)}")
        except RuntimeError as rte:
            logger.error(f"RuntimeError during processing file {display_name}: {rte}", exc_info=True)
            processed_files_responses.append(
                DocumentUploadResponse(
                    filename=display_name, message=f"Runtime error during processing: {str(rte)}", total_chunks=0
                )
            )
        except HTTPException as http_exc:  # 如果内部抛出HTTPException，直接记录并添加到响应
            logger.error(f"HTTPException for file {display_name}: {http_exc.detail}", exc_info=True)
            processed_files_responses.append(
                DocumentUploadResponse(filename=display_name, message=f"Error: {http_exc.detail}", total_chunks=0)
            )
        except Exception as e:
            logger.error(f"Unexpected error during document upload for {display_name}: {e}", exc_info=True)
            processed_files_responses.append(
                DocumentUploadResponse(
                    filename=display_name,
                    message="An unexpected error occurred. See the server log for details.",
                    total_chunks=0,
                )
            )
        finally:
            await file.close()

    if not processed_files_responses:  # 如果没有选择任何文件或所有文件都被跳过
        raise HTTPException(status_code=400, detail="No valid .md files were processed.")

    return processed_files_responses  # 返回所有文件的处理结果列表


@app.post(
    "/query/",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    dependencies=[Depends(require_api_key)],
)
async def query_llm(request: QueryRequest, db: Session = Depends(get_db)):
    """
    接收用户问题, 执行RAG流程并返回答案.
    """  # Ensuring this is a clean, standard docstring with ASCII punctuation.
    try:
        user_question = request.question
        top_k = request.top_k
        logger.info(f"Received query: '{user_question}', top_k: {top_k}")

        # 1. 问题向量化
        logger.info("Embedding user question...")
        query_embedding = services.get_embeddings([user_question])
        if not query_embedding or not query_embedding[0]:
            logger.error("Failed to embed user question.")
            raise HTTPException(status_code=500, detail="Failed to vectorize question.")
        logger.info("User question embedded successfully.")

        # 2. Milvus 检索
        logger.info(f"Searching Milvus for top-{top_k} chunks.")
        retrieved_milvus_chunks_before_filter = services.search_in_milvus(query_embedding[0], top_k)

        # 根据距离阈值过滤检索结果
        retrieved_milvus_chunks = []
        if retrieved_milvus_chunks_before_filter:
            for chunk_info in retrieved_milvus_chunks_before_filter:
                if chunk_info["distance"] <= MILVUS_SEARCH_MAX_DISTANCE:
                    retrieved_milvus_chunks.append(chunk_info)
                else:
                    logger.info(
                        f"Chunk {chunk_info['chunk_id']} with distance {chunk_info['distance']} filtered out (threshold: {MILVUS_SEARCH_MAX_DISTANCE})."
                    )
            logger.info(
                f"{len(retrieved_milvus_chunks_before_filter) - len(retrieved_milvus_chunks)} chunks filtered out by distance threshold."
            )

        rag_source_chunks_details = []
        rag_context_text = "没有在文档中找到相关信息。"  # 默认值

        if not retrieved_milvus_chunks:
            logger.info("No relevant chunks found in Milvus. Proceeding without RAG context.")
            # rag_context_text 和 rag_source_chunks_details 使用默认值
        else:
            retrieved_chunk_ids = [chunk_info["chunk_id"] for chunk_info in retrieved_milvus_chunks]
            logger.info(f"Retrieved {len(retrieved_chunk_ids)} chunk IDs from Milvus: {retrieved_chunk_ids}")

            rag_source_db_chunks = crud.get_chunks_by_ids(db, retrieved_chunk_ids)
            if not rag_source_db_chunks:
                logger.warning("Could not retrieve chunk details from SQLite even though Milvus returned IDs.")
                rag_context_text = "检索到的信息无法加载。"
            else:
                logger.info(f"Retrieved {len(rag_source_db_chunks)} chunk contents from SQLite.")
                rag_source_chunks_details = [
                    SourceChunk(
                        chunk_id=db_chunk.chunk_id,
                        original_content=db_chunk.original_content,
                        source_document_name=db_chunk.source_document_name,
                    )
                    for db_chunk in rag_source_db_chunks
                ]
                rag_context_text = "\n\n---\n\n".join([chunk.original_content for chunk in rag_source_chunks_details])
                logger.info(f"Constructed RAG context of length {len(rag_context_text)} characters.")

        # 4. 构建 RAG Prompt
        rag_prompt = f"以下是相关背景信息：\n\n{rag_context_text}\n\n---\n基于以上信息，请回答问题：{user_question}"
        logger.info("Constructed RAG prompt.")

        # 5. 调用 LLM (RAG)
        logger.info("Getting RAG response from LLM...")
        rag_answer = services.get_llm_response(rag_prompt)
        logger.info("Received RAG response from LLM.")

        # 6. 调用 LLM (无RAG)
        logger.info("Getting non-RAG response from LLM...")
        llm_only_prompt = user_question
        llm_only_answer = services.get_llm_response(llm_only_prompt)
        logger.info("Received non-RAG response from LLM.")

        # 7. 计算相似度 (新增部分)
        similarity_scores_list: List[SimilarityScore] = []
        if rag_source_chunks_details:  # 仅当有引用来源时才计算相似度
            # 向量化 RAG answer 和 LLM-only answer
            # 注意：get_embeddings 期望一个列表，返回一个列表的列表
            rag_answer_embedding = services.get_embeddings([rag_answer])[0] if rag_answer else None
            llm_only_answer_embedding = services.get_embeddings([llm_only_answer])[0] if llm_only_answer else None

            if rag_answer_embedding and llm_only_answer_embedding:
                for source_chunk in rag_source_chunks_details:
                    source_chunk_content = source_chunk.original_content
                    source_chunk_embedding = (
                        services.get_embeddings([source_chunk_content])[0] if source_chunk_content else None
                    )

                    if source_chunk_embedding:
                        sim_with_rag = services.cosine_similarity(rag_answer_embedding, source_chunk_embedding)
                        sim_with_llm_only = services.cosine_similarity(
                            llm_only_answer_embedding, source_chunk_embedding
                        )

                        similarity_scores_list.append(
                            SimilarityScore(
                                source_chunk_id=source_chunk.chunk_id,
                                source_document_name=source_chunk.source_document_name,
                                similarity_with_rag_answer=sim_with_rag,
                                similarity_with_llm_only_answer=sim_with_llm_only,
                            )
                        )
                    else:
                        logger.warning(
                            f"Could not embed source chunk ID: {source_chunk.chunk_id}, skipping for similarity."
                        )
            else:
                logger.warning("Could not embed RAG answer or LLM-only answer, skipping similarity calculation.")
        else:
            logger.info("No RAG sources, skipping similarity calculation.")

        return QueryResponse(
            rag_answer=rag_answer,
            rag_sources=rag_source_chunks_details,
            llm_only_answer=llm_only_answer,
            similarity_scores=similarity_scores_list if similarity_scores_list else None,
        )

    except ValueError as ve:
        logger.error(f"ValueError during query processing: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except ConnectionError as ce:
        logger.error(f"ConnectionError during query processing: {ce}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Service connection error: {str(ce)}")
    except RuntimeError as rte:
        logger.error(f"RuntimeError during query processing: {rte}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Runtime error during processing: {str(rte)}")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred. See the server log for details.")


@app.get("/health")
def health_check():
    """简单的健康检查端点."""  # Ensuring this is a clean, standard docstring with ASCII punctuation.
    return {"status": "healthy"}


# Serve the bundled frontend (index.html, script.js, style.css) at "/" so that the UI and API share one origin.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if SERVE_FRONTEND and FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def reload_enabled(argv: Optional[Sequence[str]] = None, environ: Optional[dict] = None) -> bool:
    """Auto-reload for `python -m backend.main` is off unless `--reload` or SIMPLE_RAG_RELOAD is given."""
    import argparse

    from backend.config import parse_bool

    parser = argparse.ArgumentParser(prog="python -m backend.main")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="restart the server when a source file changes (development only; also SIMPLE_RAG_RELOAD=1)",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    environ = os.environ if environ is None else environ
    return args.reload or parse_bool(environ.get("SIMPLE_RAG_RELOAD"))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of `python -m backend.main [--reload]`."""
    import uvicorn

    from backend.config import FASTAPI_HOST, FASTAPI_PORT

    reload = reload_enabled(argv)
    if not is_loopback_host(FASTAPI_HOST) and not SIMPLE_RAG_API_KEY:
        raise SystemExit(
            f"Refusing to listen on non-loopback address {FASTAPI_HOST} without SIMPLE_RAG_API_KEY. "
            "Set SIMPLE_RAG_API_KEY or use FASTAPI_HOST=127.0.0.1."
        )
    # Tell the lifespan check (also in the reloader's child process) which address is used.
    os.environ[BIND_HOST_ENV] = FASTAPI_HOST
    logger.info(f"Starting Uvicorn server on {FASTAPI_HOST}:{FASTAPI_PORT} (reload={'on' if reload else 'off'})")
    uvicorn.run("backend.main:app", host=FASTAPI_HOST, port=FASTAPI_PORT, reload=reload)


if __name__ == "__main__":
    main()
