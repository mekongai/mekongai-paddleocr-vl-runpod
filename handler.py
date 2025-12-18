"""
RunPod Handler with OpenAI-Compatible API for PaddleOCR-VL

This runs a FastAPI server that exposes /v1/chat/completions endpoint,
compatible with PaddleOCRVL library's vllm-server backend.
"""
import os
import base64
import tempfile
import json
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None

def load_model():
    """Load PaddleOCR-VL model once, reuse for all requests."""
    global pipeline
    if pipeline is None:
        from paddlex import create_pipeline
        logger.info("Loading PaddleOCR-VL model...")
        pipeline = create_pipeline("PaddleOCR-VL")
        logger.info("Model loaded successfully!")
    return pipeline


# ============================================================
# Pydantic Models for OpenAI-compatible API
# ============================================================

class ImageUrl(BaseModel):
    url: str  # Can be data:image/png;base64,xxx or http URL

class ContentItem(BaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: Any  # Can be string or list of ContentItem

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.0

class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="PaddleOCR-VL OpenAI Compatible API")


@app.get("/health")
@app.get("/ping")
async def health_check():
    """Health check endpoint for RunPod"""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "PaddleOCR-VL",
                "object": "model",
                "owned_by": "paddlepaddle"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Processes images with PaddleOCR-VL.
    """
    import time
    import uuid
    
    try:
        # Extract image from messages
        image_data = None
        prompt_text = ""
        
        for message in request.messages:
            if isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                            if url.startswith("data:image"):
                                # Extract base64 from data URL
                                image_data = url.split(",", 1)[1] if "," in url else url
                        elif item.get("type") == "text":
                            prompt_text = item.get("text", "")
                    elif hasattr(item, 'type'):
                        if item.type == "image_url" and item.image_url:
                            url = item.image_url.url
                            if url.startswith("data:image"):
                                image_data = url.split(",", 1)[1] if "," in url
                        elif item.type == "text":
                            prompt_text = item.text or ""
            elif isinstance(message.content, str):
                prompt_text = message.content
        
        if not image_data:
            raise HTTPException(status_code=400, detail="No image provided in request")
        
        # Decode and save image
        image_bytes = base64.b64decode(image_data)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name
        
        try:
            # Load model and run OCR
            model = load_model()
            results = list(model.predict(temp_path))
            
            # Format output as text
            output_text = ""
            for page_result in results:
                if hasattr(page_result, 'json'):
                    page_data = page_result.json
                elif hasattr(page_result, 'to_dict'):
                    page_data = page_result.to_dict()
                elif isinstance(page_result, dict):
                    page_data = page_result
                else:
                    page_data = {"text": str(page_result)}
                
                # Extract text from OCR result
                if isinstance(page_data, dict):
                    # Try different possible formats
                    if 'text' in page_data:
                        output_text += page_data['text'] + "\n"
                    elif 'rec_texts' in page_data:
                        output_text += "\n".join(page_data['rec_texts']) + "\n"
                    elif 'blocks' in page_data:
                        for block in page_data.get('blocks', []):
                            if 'text' in block:
                                output_text += block['text'] + "\n"
                    else:
                        output_text += json.dumps(page_data) + "\n"
                else:
                    output_text += str(page_data) + "\n"
            
            # Build OpenAI-compatible response
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message={
                            "role": "assistant",
                            "content": output_text.strip()
                        },
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=100,
                    completion_tokens=len(output_text.split()),
                    total_tokens=100 + len(output_text.split())
                )
            )
            
            return response
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Get port from environment (RunPod sets this)
    port = int(os.environ.get("PORT", 8008))
    health_port = int(os.environ.get("PORT_HEALTH", port))
    
    logger.info(f"Starting OpenAI-compatible server on port {port}")
    
    # Pre-load model
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=port)
