from datetime import timedelta
import re

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    db_dialect: str
    db_driver: str
    db_username: str
    db_password: str
    db_host: str
    db_port: str
    db_database: str
    db_conn_string: str = ""
    milvus_endpoint: str
    model_config = SettingsConfigDict(env_file=".env")

_settings = Settings()

_settings.db_conn_string = f"{_settings.db_dialect}+{_settings.db_driver}://{_settings.db_username}:{_settings.db_password}@{_settings.db_host}:{_settings.db_port}/{_settings.db_database}"