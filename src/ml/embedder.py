from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from typing import List

class AudioEmbedder:
    def __init__(self, s3_client: S3Client, pg_client: PostgresClient, milvus_client: MilvusClient):
        self.s3_client = s3_client
        self.pg_client = pg_client
        self.milvus_client = milvus_client
        self.model = imagebind_huge(pretrained=True)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def load_and_transform_audio_data(self, audio_paths: List[str]) -> List[List[float]]:
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device),
        }
        with torch.no_grad():
            embeddings = self.model(inputs)

        return embeddings[ModalityType.AUDIO]

    def process(self):
        paths = self.pg_client.get_audio_paths()
        local_paths = self.s3_client.download_files(paths, '/tmp')
        embeddings = self.load_and_transform_audio_data(local_paths)
        self.milvus_client.insert_embeddings(embeddings)
