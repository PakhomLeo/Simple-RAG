import datetime
import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from . import database  # 使用相对导入


def create_chunk(db: Session, chunk_content: str, document_name: str) -> database.ChunkDB:
    chunk_id = str(uuid.uuid4())
    db_chunk = database.ChunkDB(
        chunk_id=chunk_id,
        original_content=chunk_content,
        source_document_name=document_name,
        upload_timestamp=datetime.datetime.utcnow(),
    )
    db.add(db_chunk)
    db.commit()
    db.refresh(db_chunk)
    return db_chunk


def add_chunks(db: Session, chunk_contents: List[str], document_name: str) -> List[str]:
    """Add the chunks of one document to the session without committing. Returns their ids.

    The caller commits once the vectors are in Milvus, or rolls back, so SQLite never keeps chunks
    that have no vector."""
    now = datetime.datetime.utcnow()
    ids = []
    for content in chunk_contents:
        chunk_id = str(uuid.uuid4())
        db.add(
            database.ChunkDB(
                chunk_id=chunk_id, original_content=content, source_document_name=document_name, upload_timestamp=now
            )
        )
        ids.append(chunk_id)
    db.flush()
    return ids


def get_chunk_by_id(db: Session, chunk_id: str) -> Optional[database.ChunkDB]:
    return db.query(database.ChunkDB).filter(database.ChunkDB.chunk_id == chunk_id).first()


def get_chunks_by_ids(db: Session, chunk_ids: List[str]) -> List[database.ChunkDB]:
    if not chunk_ids:
        return []
    return db.query(database.ChunkDB).filter(database.ChunkDB.chunk_id.in_(chunk_ids)).all()


def get_all_chunks(db: Session, skip: int = 0, limit: int = 100) -> List[database.ChunkDB]:
    return db.query(database.ChunkDB).offset(skip).limit(limit).all()


def delete_all_chunks(db: Session) -> int:
    """Delete every stored chunk. Returns the number of deleted rows."""
    deleted = db.query(database.ChunkDB).delete(synchronize_session=False)
    db.commit()
    return deleted
