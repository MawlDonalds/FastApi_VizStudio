# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# API KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyC1U2KrsHXvaS1Vx73rCcR9KPn70qCzdyU")

# Database Credentials
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin123")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "sales")

CONNECTION_STRING = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"