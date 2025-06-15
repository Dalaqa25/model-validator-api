from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
from pydantic import BaseModel
import zipfile
import io
import os
from typing import List, Dict
import mimetypes
import httpx
import tempfile
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek/deepseek-r1-0528-qwen3-8b:free")
SITE_URL = os.getenv("SITE_URL", "http://localhost:3000")
SITE_NAME = os.getenv("SITE_NAME", "Model Validator API")

app = FastAPI()

class AIRequest(BaseModel):
    prompt: str

async def get_ai_analysis(content: str) -> str:
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME
        }
        
        data = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze this code and provide insights about what it does, its purpose, and any potential improvements. Here's the code:\n\n{content}"
                }
            ]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error: API returned status code {response.status_code}"
            
    except Exception as e:
        return f"Error in AI analysis: {str(e)}"

@app.get("/")
def root():
    return {"message": "FastAPI is running!"}

@app.post("/ai")
async def get_ai_response(request: AIRequest):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME
        }
        
        data = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return {"response": result['choices'][0]['message']['content']}
            else:
                return {"error": f"API returned status code {response.status_code}"}
            
    except Exception as e:
        return {"error": f"Error in AI response: {str(e)}"}

async def analyze_file_content(file_path: str) -> Dict:
    """Analyze the content of a file and return its details."""
    try:
        # Skip hidden files
        if os.path.basename(file_path).startswith('._'):
            return {
                "file_name": os.path.basename(file_path),
                "file_type": "hidden",
                "file_size": f"{os.path.getsize(file_path)} bytes",
                "content": "Hidden system file",
                "ai_analysis": ""
            }
            
        # Get file type
        mime_type, _ = mimetypes.guess_type(file_path)
        file_type = mime_type if mime_type else 'unknown'
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Read file content based on type
        content = ""
        ai_analysis = ""
        
        if file_type.startswith('text/'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Get AI analysis for text files
            if file_type == 'text/x-python' or file_type == 'text/plain':
                ai_analysis = await get_ai_analysis(content)
        elif file_type.startswith('image/'):
            content = "Binary image file"
        else:
            content = "Binary file"
            
        return {
            "file_name": os.path.basename(file_path),
            "file_type": file_type,
            "file_size": f"{file_size} bytes",
            "content": content[:1000] + "..." if len(content) > 1000 else content,
            "ai_analysis": ai_analysis
        }
    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "error": f"Error analyzing file: {str(e)}"
        }

@app.post("/process-zip")
async def process_zip_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        return {"error": "File must be a ZIP file"}
    
    # Create a unique temporary directory
    extract_dir = tempfile.mkdtemp(prefix="zip_analysis_")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Create a BytesIO object from the contents
        zip_file = io.BytesIO(contents)
        
        # Extract the ZIP file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Analyze each extracted file
        file_analyses = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                file_path = os.path.join(root, file)
                analysis = await analyze_file_content(file_path)
                file_analyses.append(analysis)
        
        return {
            "message": "ZIP file processed successfully",
            "files_analyzed": file_analyses
        }
        
    except Exception as e:
        return {"error": f"Error processing ZIP file: {str(e)}"}
    finally:
        # Clean up: remove the temporary directory and its contents
        try:
            shutil.rmtree(extract_dir)
        except Exception as e:
            print(f"Error cleaning up temporary directory: {str(e)}")