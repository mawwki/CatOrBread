"""FastAPI backend for Cat or Bread classifier"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from model.predict import predict

app = FastAPI(title="Cat or Bread?")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Cat or Bread?</h1><p>Frontend not found</p>")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict(contents)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    return JSONResponse(result)

@app.get("/health")
async def health():
    from model.predict import get_model
    model = get_model()
    return {"status": "ok" if model else "no_model"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
