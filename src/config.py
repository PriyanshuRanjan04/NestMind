"""
Application configuration — loaded once at import time.
All env vars are documented in .env.example.
"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
APP_ENV: str = os.getenv("APP_ENV", "development")

# Pipeline hard timeout (seconds). Chosen to stay comfortably under the 6 s
# p95 end-to-end target while still allowing a slow first token.
PIPELINE_TIMEOUT_S: float = float(os.getenv("PIPELINE_TIMEOUT_S", "25"))

# Classifier retry budget — one retry on transient OpenAI errors before we
# fall back to a safe default.
CLASSIFIER_MAX_RETRIES: int = int(os.getenv("CLASSIFIER_MAX_RETRIES", "1"))
