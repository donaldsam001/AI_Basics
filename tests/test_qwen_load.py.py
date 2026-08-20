import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env variables
load_dotenv()

model_path = os.getenv("QWEN_MODEL_PATH")
is_enabled = os.getenv("QWEN_ENABLED")

print(f"1. Env QWEN_ENABLED: {is_enabled}")
print(f"2. Env QWEN_MODEL_PATH: {model_path}")

if not model_path:
    print("❌ ERROR: QWEN_MODEL_PATH is not set or .env is not being read.")
    exit(1)

path_obj = Path(model_path)
print(f"3. Absolute Path: {path_obj.resolve()}")
print(f"4. File Exists: {path_obj.is_file()}")

if path_obj.is_file():
    print("5. Attempting to load model into memory...")
    try:
        from src.llm import QwenLLM  # Adjust import based on your exact structure
        # If your init takes a model_path parameter:
        llm = QwenLLM(model_path=model_path)
        print("✅ SUCCESS: Model loaded perfectly!")
    except Exception as e:
        print(f"❌ ERROR LOADING MODEL: {e}")