from pydantic import BaseModel
from sqlalchemy.orm import Session
import database.db_models as db_models


class Text_record(BaseModel):
    text: str
    sentiment: int
    probability: float
    model: str

class Batch_record(BaseModel):
    is_done: bool
    texts: list[Text_record]



class DB:
    def __init__(self):
        self.db = db_models.SessionLocal()
    
    def add_batch_req(self, batch_record: Batch_record):
        # -- insert batch for its Id
        db_batch_req = db_models.Batch_Request(is_done=batch_record.is_done)
        self.db.add(db_batch_req)
        self.db.commit()
        self.db.refresh(db_batch_req)

        # -- insert texts
        for t in batch_record.texts:
            db_text = None
            if batch_record.is_done:
                db_text = db_models.Text(
                    text=t.text, 
                    sentiment=t.sentiment,
                    probability=t.probability,
                    model=t.model,
                    batch_id=db_batch_req.id
                    )
            else:
                db_text = db_models.Text(
                    text=t.text,
                    batch_id=db_batch_req.id
                    )
            self.db.add(db_text)
        
        self.db.commit()

        return db_batch_req

    def add_text(self, text_record: Text_record):
        db_text = db_models.Text(
            text=text_record.text, 
            sentiment=text_record.sentiment,
            probability=text_record.probability,
            model=text_record.model
            )
        self.db.add(db_text)
        self.db.commit()
        self.db.refresh(db_text)
        return db_text


    async def get_texts(self, skip: int = 0, limit: int = 100, model: str = "A&B", text=""):
        if model == "A&B":
            if text == "":
                return self.db.query(db_models.Text).offset(skip).limit(limit).all()
            else:
                return self.db.query(db_models.Text).filter(db_models.Text.text.contains(text)).offset(skip).limit(limit).all()
        else:
            if text == "":
                return self.db.query(db_models.Text).where(db_models.Text.model == model).offset(skip).limit(limit).all()
            else:
                return self.db.query(db_models.Text).where(db_models.Text.model == model).filter(db_models.Text.text.contains(text)).offset(skip).limit(limit).all()


    async def get_batch_req(self, skip: int = 0, limit: int = 100):
        return self.db.query(db_models.Batch_Request).offset(skip).limit(limit).all()


    async def get_batch_not_done(self):
        return self.db.query(db_models.Batch_Request).where(db_models.Batch_Request.is_done == False).order_by(db_models.Batch_Request.created_at).first()
