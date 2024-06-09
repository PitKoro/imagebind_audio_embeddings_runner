import logging
import time

from runners.simple_runner import SimpleRunner
from storage.postgrya import PostgresClient
from storage.milfus import CustomMilvusClient
from settings import _settings

def main():
    logging.basicConfig(
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%d-%m-%y %H:%M:%S',
        level='INFO',
    )
    logger = logging.getLogger('Runner')
    
    logger.info(f'Connecting to postgres DB...')
    pg_client = PostgresClient(_settings.db_conn_string)
    logger.info(f'Connecting to Milfus...')
    milvus_client = CustomMilvusClient(
        milvus_endpoint=_settings.milvus_endpoint,
        db_name="image_bind"
    )

    audio_embedder = AudioEmbedder()

    runner = SimpleRunner(pg_client, milvus_client, logger, seed_urls)
    start = time.time()
    runner.run()
    logger.info(f'Total duration is {time.time() - start}')
                

if __name__ == '__main__':
    main()