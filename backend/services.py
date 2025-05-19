import requests
import json
from typing import List, Dict, Any
from backend.config import (
    EMBEDDING_API_URL,
    ARK_API_KEY,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION_NAME,
    MILVUS_DIMENSION,
    MILVUS_ID_FIELD_NAME,
    MILVUS_VECTOR_FIELD_NAME,
    OLLAMA_BASE_URL,
    LLM_MODEL_NAME,
    MILVUS_METRIC_TYPE,
    MILVUS_INDEX_TYPE,
    MILVUS_INDEX_PARAM_NLIST,
    MILVUS_SEARCH_PARAM_NPROBE,
    MILVUS_CONSISTENCY_LEVEL
)
from pymilvus import connections, utility, Collection, DataType, FieldSchema, CollectionSchema
import ollama
import logging
import os
import re # <--- 添加导入 re 模块
import numpy as np # <--- 添加导入 numpy 用于向量运算

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Text Processing ---
def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    将输入文本按指定的块大小和重叠量切分为多个文本块。

    Args:
        text: 待切分的原始文本。
        chunk_size: 每个文本块的目标大小（字符数）。
        chunk_overlap: 相邻文本块之间的重叠大小（字符数），以保证上下文连续性。

    Returns:
        一个包含切分后文本块的列表。
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start += (chunk_size - chunk_overlap)
        if start >= text_len: 
             break
    return chunks

# --- Embedding Service ---
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    调用火山方舟 Embedding API 将一批文本转换为向量表示。

    Args:
        texts: 需要转换为向量的文本字符串列表。

    Returns:
        一个包含对应文本向量的列表，每个向量是一个浮点数列表。

    Raises:
        ValueError: 如果API请求失败或返回数据格式不正确。
    """
    if not texts:
        return []

    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "input": texts,
        "encoding_format": "float",
    }
    
    logger.info(f"Requesting embeddings for {len(texts)} text snippets from {EMBEDDING_API_URL}")
    
    try:
        response = requests.post(EMBEDDING_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        
        if "data" not in response_data or not isinstance(response_data["data"], list):
            logger.error(f"Unexpected response format from embedding API: {response_data}")
            raise ValueError("Embedding API response missing 'data' list.")

        embeddings = []
        for item in response_data["data"]:
            if "embedding" not in item or not isinstance(item["embedding"], list):
                logger.error(f"Embedding item missing 'embedding' list: {item}")
                raise ValueError("Invalid embedding item format.")
            embeddings.append(item["embedding"])
        
        logger.info(f"Successfully retrieved {len(embeddings)} embeddings.")
        # 校验第一个embedding的维度，如果需要的话
        if embeddings and MILVUS_DIMENSION != len(embeddings[0]):
            logger.warning(f"Milvus dimension ({MILVUS_DIMENSION}) might not match actual embedding dimension ({len(embeddings[0])}).")
            # 可以选择在这里抛出错误或动态调整MILVUS_DIMENSION，但后者更复杂
            # raise ValueError(f"Embedding dimension mismatch: Expected {MILVUS_DIMENSION}, Got {len(embeddings[0])}")


        return embeddings
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Embedding API: {e}")
        logger.error(f"Response content: {response.text if response else 'No response'}")
        raise ValueError(f"Embedding API request failed: {e}")
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Error processing embedding API response: {e}")
        logger.error(f"Response content: {response_data if 'response_data' in locals() else 'Unknown response data'}")
        raise ValueError(f"Invalid response from Embedding API: {e}")

# --- Cosine Similarity --- # <--- 新增区域
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算两个向量之间的余弦相似度。"""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        logger.warning("Cannot compute cosine similarity for invalid or mismatched vectors.")
        return 0.0 

    # 转换为numpy数组以方便计算
    np_vec_a = np.array(vec_a)
    np_vec_b = np.array(vec_b)

    dot_product = np.dot(np_vec_a, np_vec_b)
    norm_a = np.linalg.norm(np_vec_a)
    norm_b = np.linalg.norm(np_vec_b)

    if norm_a == 0 or norm_b == 0:
        logger.warning("Cannot compute cosine similarity with zero vector(s).")
        return 0.0 # 避免除以零
    
    similarity = dot_product / (norm_a * norm_b)
    return float(similarity)

# --- Milvus Service ---
# 全局变量，用于缓存Milvus Collection对象，避免重复初始化
_milvus_collection = None

def _get_milvus_connection_alias() -> str:
    """为当前进程生成一个唯一的Milvus连接别名。
    这在多进程或多线程环境下（例如某些WSGI服务器）可能有助于避免连接冲突或管理问题。
    """
    return f"milvus_conn_{os.getpid()}_services" # 添加后缀以区分其他可能的连接

def connect_to_milvus():
    """确保与Milvus服务建立连接。
    如果使用指定别名的连接已存在，则不进行任何操作。
    """
    alias = _get_milvus_connection_alias()
    try:
        # 检查是否已存在具有此别名的连接
        if not connections.has_connection(alias):
            logger.info(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT} with alias {alias}")
            # 使用配置中的主机和端口进行连接
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, alias=alias)
            logger.info("Successfully connected to Milvus.")
        else:
            logger.info(f"Already connected to Milvus with alias {alias}.")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        raise ConnectionError(f"Milvus connection failed: {e}")

def get_milvus_collection() -> Collection:
    """获取 Milvus Collection 实例。

    如果指定的集合已存在，则加载它。
    如果不存在，则根据配置参数创建一个新的集合，并设置相应的 schema 和索引。
    函数会尝试重用全局缓存的 `_milvus_collection` 对象以提高效率。

    Returns:
        Milvus Collection 对象。
    """
    global _milvus_collection
    alias = _get_milvus_connection_alias()
    connect_to_milvus() # 确保与Milvus的连接已建立

    # 尝试使用缓存的Collection对象
    if _milvus_collection and _milvus_collection.name == MILVUS_COLLECTION_NAME:
        try:
            # 验证缓存的collection是否仍然存在于服务器上且有效
            if utility.has_collection(MILVUS_COLLECTION_NAME, using=alias):
                if _milvus_collection.name == MILVUS_COLLECTION_NAME: # 双重检查名称
                    # logger.debug(f"Reusing existing Milvus collection object for '{MILVUS_COLLECTION_NAME}'.")
                    return _milvus_collection
            # 如果collection在服务器上不存在或全局变量失效，则清除缓存以便重新加载
            _milvus_collection = None 
        except Exception as e: # 处理检查过程中可能发生的异常
            logger.warning(f"Error checking cached collection, attempting to re-initialize: {e}")
            _milvus_collection = None # 重置缓存

    # 如果缓存不可用或无效，则从服务器检查或创建Collection
    if utility.has_collection(MILVUS_COLLECTION_NAME, using=alias):
        logger.info(f"Collection '{MILVUS_COLLECTION_NAME}' already exists. Loading...")
        _milvus_collection = Collection(MILVUS_COLLECTION_NAME, using=alias)
        _milvus_collection.load() # 确保集合数据加载到内存中以供搜索
        logger.info(f"Collection '{MILVUS_COLLECTION_NAME}' loaded.")
        return _milvus_collection
    else:
        logger.info(f"Collection '{MILVUS_COLLECTION_NAME}' does not exist. Creating...")
        # 定义集合的 Schema
        fields = [
            FieldSchema(name=MILVUS_ID_FIELD_NAME, dtype=DataType.VARCHAR, is_primary=True, max_length=36), # 主键，通常用于存储 chunk_id (UUID4长度为36)
            FieldSchema(name=MILVUS_VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=MILVUS_DIMENSION) # 存储文本向量
        ]
        schema = CollectionSchema(fields, description="RAG Collection for text embeddings")
        
        # 创建集合
        _milvus_collection = Collection(MILVUS_COLLECTION_NAME, schema=schema, using=alias)
        logger.info(f"Collection '{MILVUS_COLLECTION_NAME}' created.")

        # 为向量字段创建索引，这对于高效搜索至关重要
        index_params = {
            "metric_type": MILVUS_METRIC_TYPE,      # 距离度量类型，如 L2 (欧氏距离) 或 IP (内积)
            "index_type": MILVUS_INDEX_TYPE,        # 索引类型，如 IVF_FLAT, HNSW 等
            "params": {"nlist": MILVUS_INDEX_PARAM_NLIST}, # 特定索引类型的参数，例如nlist (IVF_FLAT的聚类中心数)
        }
        _milvus_collection.create_index(field_name=MILVUS_VECTOR_FIELD_NAME, index_params=index_params)
        logger.info(f"Index created for field '{MILVUS_VECTOR_FIELD_NAME}' with type '{MILVUS_INDEX_TYPE}'.")
        _milvus_collection.load() # 新创建的集合也需要加载才能进行搜索
        logger.info(f"Collection '{MILVUS_COLLECTION_NAME}' loaded after creation and indexing.")
        return _milvus_collection

def insert_into_milvus(chunk_ids: List[str], embeddings: List[List[float]]):
    """将一批文本块的ID及其对应的向量插入到Milvus集合中。

    Args:
        chunk_ids: 文本块ID的列表。
        embeddings: 文本块向量的列表，顺序与chunk_ids对应。

    Raises:
        RuntimeError: 如果插入数据到Milvus失败。
    """
    if not chunk_ids or not embeddings or len(chunk_ids) != len(embeddings):
        logger.error("Mismatch between chunk_ids and embeddings or empty lists provided for Milvus insertion.")
        # 考虑是否应该抛出异常或只是记录错误并返回
        return
    
    collection = get_milvus_collection() # 获取Milvus Collection实例
    alias = _get_milvus_connection_alias()
    
    # 准备要插入的实体数据，格式为字典列表
    entities = [
        {MILVUS_ID_FIELD_NAME: chunk_id, MILVUS_VECTOR_FIELD_NAME: embedding}
        for chunk_id, embedding in zip(chunk_ids, embeddings)
    ]
    
    try:
        logger.info(f"Inserting {len(entities)} entities into Milvus collection '{collection.name}'.")
        insert_result = collection.insert(entities)
        logger.info(f"Successfully inserted entities. Primary keys (first few): {insert_result.primary_keys[:5]}...")
        # flush操作确保数据从内存刷新到磁盘，使其对搜索可见（取决于一致性级别）
        collection.flush(using=alias) 
        logger.info("Milvus data flushed.")
    except Exception as e:
        logger.error(f"Error inserting data into Milvus: {e}")
        raise RuntimeError(f"Milvus insertion failed: {e}")

def search_in_milvus(query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    """使用给定的查询向量在Milvus集合中执行相似性搜索。

    Args:
        query_embedding: 用于查询的文本向量。
        top_k: 希望检索到的最相似结果的数量。

    Returns:
        一个字典列表，每个字典包含检索到的文本块的chunk_id和与查询向量的距离。

    Raises:
        RuntimeError: 如果Milvus搜索失败。
    """
    collection = get_milvus_collection() # 获取Milvus Collection实例
    alias = _get_milvus_connection_alias()

    # 定义搜索参数
    search_params = {
        "metric_type": MILVUS_METRIC_TYPE, # 搜索时使用的距离度量类型，应与索引时一致
        "params": {"nprobe": MILVUS_SEARCH_PARAM_NPROBE}, # 特定索引类型的搜索参数，例如nprobe (影响IVF系列索引的搜索范围和精度)
    }
    
    try:
        logger.info(f"Searching in Milvus collection '{collection.name}' with top_k={top_k} using metric '{MILVUS_METRIC_TYPE}'.")
        # 执行搜索操作
        results = collection.search(
            data=[query_embedding],                       # 查询向量列表（此处为单个向量）
            anns_field=MILVUS_VECTOR_FIELD_NAME,        # 要在哪个向量字段上进行搜索
            param=search_params,                        # 搜索参数
            limit=top_k,                                # 返回结果数量上限
            expr=None,                                  # 可选的标量字段过滤表达式
            output_fields=[MILVUS_ID_FIELD_NAME],       # 指定搜索结果中需要返回的字段（除了距离和主键ID）
            consistency_level=MILVUS_CONSISTENCY_LEVEL, # 数据一致性级别
            using=alias
        )
        
        retrieved_chunks = []
        # 处理搜索结果
        if results and results[0]: # results[0] 对应第一个查询向量的搜索结果
            for hit in results[0]: # 遍历每个命中的结果
                retrieved_chunks.append({
                    "chunk_id": hit.id,         # 命中文档的主键ID (即chunk_id)
                    "distance": hit.distance    # 与查询向量的计算距离
                })
        logger.info(f"Milvus search found {len(retrieved_chunks)} results for top_k={top_k}.")
        return retrieved_chunks
    except Exception as e:
        logger.error(f"Error searching in Milvus: {e}")
        raise RuntimeError(f"Milvus search failed: {e}")

def clear_milvus_data():
    """清空 Milvus 中指定集合的数据。
    当前实现方式是直接删除（drop）该集合。
    下次调用 get_milvus_collection() 时会自动重新创建空集合。
    """
    alias = _get_milvus_connection_alias()
    connect_to_milvus() # 确保连接已建立

    collection_name = MILVUS_COLLECTION_NAME
    try:
        if utility.has_collection(collection_name, using=alias):
            logger.info(f"Milvus collection '{collection_name}' found. Attempting to drop it...")
            utility.drop_collection(collection_name, using=alias)
            logger.info(f"Successfully dropped Milvus collection '{collection_name}'.")
            
            # 重置全局缓存的collection对象，因为它现在无效了
            global _milvus_collection
            _milvus_collection = None
        else:
            logger.info(f"Milvus collection '{collection_name}' not found. No action needed for clearing.")
    except Exception as e:
        logger.error(f"Error occurred while trying to drop Milvus collection '{collection_name}': {e}")
        # 根据需求，这里可以决定是否抛出异常，如果清空失败是关键错误的话
        # raise RuntimeError(f"Failed to clear Milvus collection '{collection_name}': {e}")

# --- Ollama LLM 服务 ---
def get_llm_response(prompt: str) -> str:
    """调用本地 Ollama 服务中的 LLM (由配置指定) 生成对给定 prompt 的响应。

    Args:
        prompt: 输入给 LLM 的提示文本。

    Returns:
        LLM 生成的文本响应。

    Raises:
        RuntimeError: 如果调用 Ollama LLM 失败。
    """
    try:
        logger.info(f"Sending prompt to Ollama model {LLM_MODEL_NAME} at {OLLAMA_BASE_URL}.")
        # 可以取消注释以下行以调试prompt内容（注意可能暴露敏感信息）
        # logger.debug(f"Prompt content (first 500 chars): {prompt[:500]}...")
        
        # 初始化Ollama客户端，指定Ollama服务的基础URL
        client = ollama.Client(host=OLLAMA_BASE_URL)
        # 调用chat接口进行对话式生成
        response = client.chat(
            model=LLM_MODEL_NAME, # 使用配置中指定的模型名称
            messages=[{'role': 'user', 'content': prompt}] # 构建消息列表
        )
        
        # 从响应中提取LLM生成的答案内容
        answer = response['message']['content']
        
        # 移除 <think>...</think> 标签及其内容
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        answer = answer.strip() # 移除可能因替换产生的首尾空白

        logger.info("Successfully received response from Ollama.")
        # 可以取消注释以下行以调试响应内容
        # logger.debug(f"Ollama response (first 200 chars): {answer[:200]}...")
        return answer
    except Exception as e: # 捕获所有可能的Ollama客户端或请求异常
        logger.error(f"Error calling Ollama LLM: {e}")
        # 尝试记录更详细的错误响应（如果可用）
        if hasattr(e, 'response') and e.response is not None:
            try:
                logger.error(f"Ollama error response details: {e.response.text}")
            except Exception:
                pass # 如果响应不是文本或无法读取，则忽略
        raise RuntimeError(f"Ollama LLM generation failed: {e}")

# --- Markdown 解析服务 ---
from markdown_it import MarkdownIt # 导入Markdown解析库

def parse_markdown_file(file_content: str) -> str:
    """
    使用 markdown-it-py 解析 Markdown 文件内容，并提取纯文本。
    当前实现主要提取文本节点和代码块内容。

    Args:
        file_content: Markdown格式的字符串内容。

    Returns:
        从Markdown中提取的纯文本字符串，不同块由换行符连接。
    """
    md = MarkdownIt()
    # 解析Markdown内容为token流
    tokens = md.parse(file_content)
    plain_text_parts = []
    # 遍历token，提取文本内容
    for token in tokens:
        # 处理普通文本和内联元素中的文本
        if token.type == 'text' or token.type == 'inline':
            if hasattr(token, 'content') and token.content:
                plain_text_parts.append(token.content)
        # 处理代码块，保留其原始内容
        elif token.type == 'fence' and token.tag == 'code': 
             if hasattr(token, 'content') and token.content:
                plain_text_parts.append(token.content) 
    
    # 将提取的文本部分用换行符连接，以保留一定的原始结构感
    full_text = "\n".join(plain_text_parts)
    # logger.debug(f"Parsed markdown, extracted text length: {len(full_text)}")
    return full_text

# --- 服务初始化辅助函数 ---
def initialize_services():
    """在应用启动时初始化并检查核心外部服务 (如 Milvus, Ollama) 的可用性。
    这有助于在应用早期发现配置或连接问题。
    """
    logger.info("Initializing services...")
    
    # 初始化 Milvus 连接和集合
    try:
        connect_to_milvus()      # 尝试连接到Milvus
        get_milvus_collection()  # 尝试获取或创建主数据集合
        logger.info("Milvus connection and collection setup successfully checked/initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Milvus during startup: {e}")
        # 根据实际应用的健壮性需求，这里可以选择是否因Milvus初始化失败而阻止应用启动
        # 例如：raise RuntimeError(f"Critical service initialization failed: Milvus - {e}")

    # 检查 Ollama 服务是否可达
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        client.list() # 发送一个简单的API请求 (如列出模型) 来测试连接性
        logger.info(f"Ollama service at {OLLAMA_BASE_URL} is reachable and responsive.")
    except Exception as e:
        # Ollama服务不可达是一个警告，因为某些情况下可能允许应用在无LLM的情况下运行部分功能
        logger.warning(f"Could not connect to Ollama service at {OLLAMA_BASE_URL}: {e}. Ensure Ollama is running.")

# 确保os模块已导入（如果之前没有全局导入），因为_get_milvus_connection_alias中用到了os.getpid()
# (在当前文件结构中，os 已在顶部导入)
# import os 