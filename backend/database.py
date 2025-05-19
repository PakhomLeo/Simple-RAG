from sqlalchemy import create_engine, Column, String, Text, DateTime, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.config import SQLITE_DATABASE_URL
import datetime

engine = create_engine(SQLITE_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

metadata = MetaData()

class ChunkDB(Base):
    __tablename__ = "chunks"

    chunk_id = Column(String, primary_key=True, index=True)
    original_content = Column(Text, nullable=False)
    source_document_name = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    # 可选: document_title = Column(String, nullable=True)

def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 