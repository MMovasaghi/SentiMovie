from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship


SQLALCHEMY_DATABASE_URL = "postgresql://postgres:yCiLq5M38YewIwwtvUtjmBu4rnOG7ftb@my-postgres.mlsd-sentimovie-test.svc:5432"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Text(Base):
    __tablename__ = "texts"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    sentiment = Column(Integer, default=None) # 0: Negative, 1: Positive
    probability = Column(Float, default=None) # probability of sentiment
    model = Column(String, default="A")
    batch_id = Column(Integer, ForeignKey("batch_requests.id"), default=None)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=None)
    batch_request = relationship("Batch_Request", back_populates="batch_texts")


class Batch_Request(Base):
    __tablename__ = "batch_requests"
    id = Column(Integer, primary_key=True, index=True)
    is_done = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=None)
    batch_texts = relationship("Text", back_populates="batch_request")