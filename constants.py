import os

from dotenv import load_dotenv

# Charge les fichiers .env
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
"""Clé d'API Groq"""