from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set or couldn't be loaded")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek/deepseek-r1-0528-qwen3-8b:free")
SITE_URL = os.getenv("SITE_URL", "http://localhost:3000")
SITE_NAME = os.getenv("SITE_NAME", "Model Validator API")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AIRequest(BaseModel):
    prompt: str

async def get_ai_analysis(content: str, description: str = "", setup: str = "") -> str:
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
                    "content": f"""Analyze the following code and documentation for a machine learning model. Your task is to determine if the model should be PUBLISHED or REJECTED based on these criteria:

1. Code Quality and Relevance:
   - The code must implement the functionality described in the model description
   - The code must be relevant to the stated purpose
   - Even if the code is simple or minimal, it should be accepted as long as it works
   - Accept any working code, even if it's basic or uses common libraries

2. Documentation Quality:
   - The description must clearly explain the model's purpose
   - The setup instructions must be detailed and match the code
   - Both must be more than just repeated statements

3. Consistency:
   - The code, description, and setup instructions must all align
   - There should be no contradictions between them

Model Description: {description}

Setup Instructions: {setup}

Code:
{content}

Based on these criteria, provide a detailed analysis and end with either '✅ PUBLISH' or '❌ REJECT'. Be very lenient in your evaluation - if the code works and matches the description, it should be published even if it's simple or uses common libraries. Only reject if the code is completely non-functional or has no relation to the description."""
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

async def analyze_file_content(file_path: str, description: str = "", setup: str = "") -> Dict:
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
                ai_analysis = await get_ai_analysis(content, description, setup)
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
async def process_zip_file(
    file: UploadFile = File(...),
    description: str = Form(...),
    setup: str = Form(...)
):
    if not file.filename.endswith('.zip'):
        return {
            "isValid": False,
            "message": "File must be a ZIP file",
            "files_analyzed": [],
            "ai_analysis": None
        }
    
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
        has_python_files = False
        all_python_content = []
        
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                file_path = os.path.join(root, file)
                analysis = await analyze_file_content(file_path, description=description, setup=setup)
                file_analyses.append(analysis)
                
                # Check for Python files
                if file.endswith('.py'):
                    has_python_files = True
                    if 'content' in analysis:
                        all_python_content.append(analysis['content'])
        
        # Basic validation rules
        validation_message = []
        if not has_python_files:
            validation_message.append("No Python files found in the ZIP")
        
        # Get AI analysis of all Python files combined
        ai_analysis = None
        if all_python_content:
            combined_content = "\n\n".join(all_python_content)
            ai_analysis = await get_ai_analysis(combined_content, description, setup)
            
            # Only reject if AI explicitly says to reject
            is_rejected = "❌ REJECT" in ai_analysis
            if is_rejected:
                validation_message.append("Code appears to be a placeholder or test code")
        
        # Determine if the validation passed
        is_valid = has_python_files and not validation_message
        
        # Clean up the temporary directory
        shutil.rmtree(extract_dir)
        
        return {
            "isValid": is_valid,
            "message": "Validation successful" if is_valid else "Validation failed: " + "; ".join(validation_message),
            "files_analyzed": file_analyses,
            "ai_analysis": ai_analysis
        }
        
    except Exception as e:
        # Clean up the temporary directory in case of error
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        raise HTTPException(status_code=500, detail=str(e)) 