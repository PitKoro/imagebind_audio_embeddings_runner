import logging
import time

from runners.audio_runner import AudioRunner
from storage.postgrya import PostgresClient
from storage.milfus import AudioImageBindMilvusClient
from ml.embedder import ImageBindModel
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
    milvus_client = AudioImageBindMilvusClient(
        milvus_endpoint=_settings.milvus_endpoint,
        db_name="image_bind"
    )
    logger.info(f'Initializing imagebind model...')
    model  = ImageBindModel()
    logger.info(f'Initializing runner...')
    runner = AudioRunner(pg_client, milvus_client, logger, model)
    start = time.time()
    runner.run(batch_size=128)
    logger.info(f'Total duration is {time.time() - start}')


if __name__ == '__main__':
    main()