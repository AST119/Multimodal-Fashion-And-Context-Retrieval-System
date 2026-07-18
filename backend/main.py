"""
FastAPI Server connecting the vector store, indexer, and query retriever.
Includes absolute PYTHONPATH environment injection to resolve local package imports during indexing.
"""
import os
import subprocess
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from retriever.search import search_fashion_database
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Semantic Fashion Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static files from the frontend folder
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.post("/index")
def trigger_indexing():
    """
    Triggers the indexing pipeline locally using Python execution.
    Injects absolute project root into PYTHONPATH to prevent ModuleNotFound errors.
    """
    try:
        print("[API] Triggering indexing process...")
        script_path = os.path.abspath("indexer/build_index.py")
        
        # Inject absolute project root to ensure absolute package imports resolve correctly
        env = os.environ.copy()
        project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        env["PYTHONPATH"] = project_root
        
        result = subprocess.run(
            ["python", script_path], 
            env=env,
            capture_output=True, 
            text=True, 
            check=True
        )
        return {"status": "success", "output": result.stdout[-1500:]}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Index pipeline script threw error: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def run_search(q: str = Query(..., min_length=1), k: int = 10):
    """Fetches matched looks using hybrid semantic parsing."""
    try:
        results = search_fashion_database(q, k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image")
def get_image(path: str):
    """Serves raw image catalog assets securely."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Requested fashion asset not found.")
    return FileResponse(path)

@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")