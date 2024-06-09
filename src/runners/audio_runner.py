import os
import logging
import shutil
from typing import List
import requests
from retry import retry
from pydub import AudioSegment
from storage.postgrya import PostgresClient, Audio
from storage.milfus import AudioImageBindMilvusClient
from ml.embedder import ImageBindModel

class AudioRunner:
    def __init__(
        self, postgres_client: PostgresClient, milvus_client: AudioImageBindMilvusClient, logger: logging.Logger, image_bind_model: ImageBindModel
    ):
        self._logger = logger
        self.postgres_client = postgres_client
        self.milvus_client = milvus_client
        self.image_bind_model = image_bind_model
        
    def _get_unique_audios(self) -> List[Audio]:
        return self.postgres_client.get_mp3_audio_by_uniq_public_links()
    

    
    def convert_to_wav(self, file_paths: List[str])->List[str]:
        output_files = []

        for file_path in file_paths:
            # Проверка наличия расширения .wav
            audio = AudioSegment.from_file(file_path)
            output_file = os.path.splitext(file_path)[0] + '.wav'
            audio.export(output_file, format="wav")
            output_files.append(output_file)
            output_files.append(file_path)

        return output_files
    
    def delete_audio_dir(self, audio_dir):
        if os.path.exists(audio_dir):
            for file_name in os.listdir(audio_dir):
                file_path = os.path.join(audio_dir, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        
    @retry(exceptions=Exception, tries=8, delay=1, backoff=2, max_delay=60, jitter=(0, 1))
    def download_audio_files_and_return_paths(self, audio_urls, target_dir):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        file_paths = []
        for url in audio_urls:
            file_name = url.split("/")[-1]
            target_path = os.path.join(target_dir, file_name)

            response = requests.get(url)
            response.raise_for_status()  # Проверка на успешность запроса

            with open(target_path, 'wb') as f:
                f.write(response.content)
            
            file_paths.append(os.path.abspath(target_path))
            
        return file_paths
    
    def _process(self, audio_items: List[Audio], collection_key = "test_audio_collection"):
        audio_dir = "./test_audios"
        self._logger.info(f"Downloading {len(audio_items)} audios...")
        audio_paths = self.download_audio_files_and_return_paths(
            [el.public_link for el in audio_items],
            audio_dir
        )
        audio_paths = self.convert_to_wav(audio_paths)
        self._logger.info(f"")
        embeddings = self.image_bind_model.get_audio_embeddings_by_audio_path(audio_paths)
        insert_data = [
            {
                "uuid":uuid,
                "public_link": public_link,
                "embedding": embedding
            } for uuid, public_link, embedding in zip(
                [el.uuid for el in audio_items],
                [el.public_link for el in audio_items],
                embeddings
            )
        ]
        self.milvus_client.insert_to_collection(collection_key, insert_data)
        self.delete_audio_dir(audio_dir)

    def run(self, batch_size = 10, collection_key = "test_audio_collection"):
        self._logger.info("Creating milvus collection...")
        self.milvus_client.drop_and_create_collection(collection_key)
        unique_audios = self._get_unique_audios()
        self._logger.info(f"Start processing {len(unique_audios)} audios...")
        for i in range(0, len(unique_audios), batch_size):
            # self._logger.info(f"Processing {len(unique_audios[i:i+batch_size])} audios")
            self._process(unique_audios[i:i+batch_size], collection_key)
            self._logger.info(f"Processed {i + len(unique_audios[i:i+batch_size])} audios")
        self.milvus_client.create_collection_index(collection_key)
        self.milvus_client.load_collection(
            collection_name=collection_key,
            replica_number=1 # Number of replicas to create on query nodes. Max value is 1 for Milvus Standalone, and no greater than `queryNode.replicas` for Milvus Cluster.
        )