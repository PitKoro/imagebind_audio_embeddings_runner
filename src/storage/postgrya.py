from typing import List
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Text, Integer, Column, TIMESTAMP, BINARY
from sqlalchemy.dialects.postgresql import JSONB


Base = declarative_base()

class Audio(Base):
    __tablename__ = 'media'
    __table_args__ = {"schema": "moood_me"}
    id = Column(Integer, primary_key=True)
    collection_name = Column(Text)
    conversions_disk = Column(Text)
    created_at = Column(TIMESTAMP)
    custom_properties = Column(JSONB)
    desc_blip2 = Column(Text)
    disk = Column(Text)
    embeddings = Column(JSONB)
    file_name = Column(Text)
    generated_conversions = Column(JSONB)
    manipulations = Column(JSONB)
    mime_type = Column(Text)
    model_id = Column(Integer)
    model_type = Column(Text)
    name = Column(Text)
    order_column = Column(Integer)
    public_link = Column(Text)
    responsive_images = Column(JSONB)
    size = Column(Integer)
    trash = Column(BINARY)
    updated_at = Column(TIMESTAMP)
    uuid = Column(Text)

class PostgresClient:
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string, echo=True)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def get_audio_public_links(self) -> List[Audio]:
        return [
            el.public_link for el in self.session.\
            query(Audio).filter(Audio.model_type == "App\Models\Audio").\
            distinct(Audio.public_link).all()
        ]
