
from storage.milfus import MilvusClient
from storage.postgrya import PostgresClient
from storage.s3 import S3Client

def main():
    s3_client = S3Client("your_bucket_name")
    pg_client = PostgresClient("your_host", 5432, "your_dbname", "your_user", "your_password")
    milvus_client = MilvusClient("your_milvus_host", 19530, "your_collection_name")

    audio_embedder = AudioEmbedder()

    paths = pg_client.get_audio_paths()
    local_paths = s3_client.download_files(paths, '/tmp')
    embeddings = audio_embedder.load_and_transform_audio_data(local_paths)
    milvus_client.insert_embeddings(embeddings)

if __name__ == "__main__":
    main()
