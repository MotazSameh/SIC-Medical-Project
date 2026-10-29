import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from src.main import get_multi_rag_answer
from src.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SystemInfoResponse,
    XrayResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical_rag_api")

app = FastAPI(
    title="Medical RAG Assistant API",
    description="Multi-RAG (Arabic Q&A + NICE/FDA Guidelines) medical assistant + chest X-ray classifier.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

GENERIC_ERROR_MESSAGE = (
    "حدث خطأ أثناء معالجة سؤالك. برجاء المحاولة مرة أخرى.\n"
    "An error occurred while processing your question. Please try again."
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg"}

XRAY_DISCLAIMER = (
    "هذه النتيجة من نموذج آلي للمساعدة فقط وليست تشخيصاً طبياً. استشر الطبيب.\n"
    "This is an automated model output, not a medical diagnosis. Consult a doctor."
)


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health_check():
    return HealthResponse(status="ok")


@app.get("/api/system-info", response_model=SystemInfoResponse, tags=["system"])
def system_info():
    return SystemInfoResponse()


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="الرسالة فارغة. برجاء كتابة سؤال. / Message is empty.")

    chat_history = [m.model_dump() for m in request.chat_history]

    try:
        result = get_multi_rag_answer(message, chat_history)
    except Exception:
        logger.exception("get_multi_rag_answer failed")
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MESSAGE)

    return ChatResponse(
        answer=result["answer"],
        chat_history=result["chat_history"],
        sources=result.get("sources", []),
        routing=result["routing"],
        trust_score=result.get("trust_score"),
        trust_label=result.get("trust_label"),
    )


@app.post("/api/xray", response_model=XrayResponse, tags=["xray"])
async def xray(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="ارفع صورة PNG أو JPG فقط. / Upload a PNG or JPG image.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="الملف فارغ. / Empty file.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="الصورة كبيرة (الحد 10MB). / Image too large (max 10MB).")

    try:
        # lazy import: TensorFlow ما يبطّأش تشغيل السيرفر
        from CV.xray_model import predict
        result = await run_in_threadpool(predict, data)
    except Exception:
        logger.exception("xray prediction failed")
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MESSAGE)

    return XrayResponse(**result, disclaimer=XRAY_DISCLAIMER)


for _folder in ("nice", "fda", "who"):
    _folder_path = DATA_RAW_DIR / _folder
    if _folder_path.exists():
        app.mount(f"/documents/{_folder}", StaticFiles(directory=str(_folder_path)), name=f"{_folder}_docs")
    else:
        logger.warning("Skipping /documents/%s mount — folder not found at %s", _folder, _folder_path)

_static_assets_dir = FRONTEND_DIR / "static"
if _static_assets_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_assets_dir)), name="frontend_static")


@app.get("/", tags=["frontend"])
def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not found — place index.html at frontend/index.html.",
        )
    return FileResponse(str(index_path))


# source .venv/Scripts/activate
# uvicorn src.api.main:app --reload --port 8000
#"e:/MNU/3th_2/NLP/RAG Medical Project/.venv/Scripts/python.exe" -m uvicorn src.api.main:app --reload --port 8000