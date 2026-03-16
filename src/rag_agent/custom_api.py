from fastapi import FastAPI, File, UploadFile, APIRouter
from rag_agent.rag_agent import load_pdf_bytes

app = FastAPI()
rag_router = APIRouter(prefix="/rag", tags=["rag"])

@rag_router.get("/get_ok")
async def get_ok(): return "ok"

@rag_router.post("/upload")
async def upload(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    len = await load_pdf_bytes(pdf_bytes)
    return f"Data uploaded. Total chunks: {len}"

app.include_router(rag_router)

