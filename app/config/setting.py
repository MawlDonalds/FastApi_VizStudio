import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "admin123")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "sales")

    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")

    LLM_MODEL: str = "gemini-1.5-flash"
    LLM_TEMPERATURE: float = 0.0

    CORS_ORIGINS: list[str] = ["http://localhost", "http://localhost:8000", "http://localhost:3000"] 
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

settings = Settings()