import logging

from typing import List, Dict
from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType
)


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

class CustomMilvusClient(MilvusClient):
    def __init__(
        self,
        milvus_endpoint: str,
        db_name: str
    ):
        super().__init__(
            uri=milvus_endpoint,
            db_name=db_name
        )

    def drop_and_create_collection(self, collection_key: str) -> None:
        if self._get_connection().has_collection(collection_key):
            self.drop_collection(collection_key)
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, descrition='Id', is_primary=True, auto_id=True),
            FieldSchema(name='public_link', dtype=DataType.VARCHAR, max_length=2500, descrition='Public link'),
            FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, description='Embeddings of audio', dim=1024)
        ]
        schema = CollectionSchema(fields=fields, description=f"Collection for audio embeddings from ImageBind model")
        self.create_collection(collection_key, schema=schema)
        
        

    def insert_to_collection(self, collection_key: str, insert_data: List[Dict]) -> None:
        primary_keys = self.insert(collection_name=collection_key, data=insert_data)
        assert len(primary_keys) == len(insert_data), logger.error("Inserted less objects in collection that expected")
        
    def create_collection_index(self, collection_key: str):
        index_params = self.prepare_index_params()
        index_params.add_index(
            field_name="embeddings",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="vector_index",
            params={ "nlist": 1024 }
        )
        self.create_index(
            collection_name=collection_key,
            index_params=index_params
        )

