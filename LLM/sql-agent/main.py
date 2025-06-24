import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


# Load environment variables
load_dotenv()

CHAT_MODEL = os.getenv("GEMINI_MODEL_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize LLM
llm = init_chat_model(model=CHAT_MODEL, api_key=GEMINI_API_KEY)
