from sqlalchemy.orm import Session
from . import database, models # 使用相对导入
from typing import List, Optional
import uuid
import datetime

def create_chunk(db: Session, chunk_content: str, document_name: str) -> database.ChunkDB:
    chunk_id = str(uuid.uuid4())
    db_chunk = database.ChunkDB(
        chunk_id=chunk_id,
        original_content=chunk_content,
        source_document_name=document_name,
        upload_timestamp=datetime.datetime.utcnow()
    )
    db.add(db_chunk)
    db.commit()
    db.refresh(db_chunk)
    return db_chunk

def get_chunk_by_id(db: Session, chunk_id: str) -> Optional[database.ChunkDB]:
    return db.query(database.ChunkDB).filter(database.ChunkDB.chunk_id == chunk_id).first()

def get_chunks_by_ids(db: Session, chunk_ids: List[str]) -> List[database.ChunkDB]:
    if not chunk_ids:
        return []
    return db.query(database.ChunkDB).filter(database.ChunkDB.chunk_id.in_(chunk_ids)).all()

def get_all_chunks(db: Session, skip: int = 0, limit: int = 100) -> List[database.ChunkDB]:
    return db.query(database.ChunkDB).offset(skip).limit(limit).all() 