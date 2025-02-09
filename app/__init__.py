from fastapi import FastAPI
from .config import get_settings

app = FastAPI()
settings = get_settings()