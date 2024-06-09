# imagebind_audio_embeddings_runner

Скрипт просчета imagebind эмбеддингов аудио в milvus

## dependences
```bash
conda create -n runner python=3.8 -y
conda activate runner
sudo apt-get update
sudo apt install postgresql
sudo apt-get install libpq-dev python3-dev
sudo apt -y install libgeos-dev
conda install ffmpeg -y
pip install -r requirements.txt
```