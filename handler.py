"""
RunPod Handler with OpenAI-Compatible API for PaddleOCR-VL
Simplified version with better error handling and logging
"""
import os
import sys
import base64
import tempfile
import json
import logging
import traceback
from typing import List, Dict, Any, Optional

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("=" * 60)
logger.info("Starting PaddleOCR-VL OpenAI-Compatible Server")
logger.info("=" * 60)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    logger.info("✅ FastAPI and uvicorn imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import FastAPI/uvicorn: {e}")
    sys.exit(1)

# Global pipeline instance
pipeline = None
model_loading = False
model_error = None

def load_model():
    """Load PaddleOCR-VL model once, reuse for all requests."""
    global pipeline, model_loading, model_error
    
    if pipeline is not None:
        return pipeline
    
    if model_loading:
        return None
    
    model_loading = True
    
    try:
        logger.info("Loading PaddleOCR-VL model...")
        from paddlex import create_pipeline
        pipeline = create_pipeline("PaddleOCR-VL")
        logger.info("✅ Model loaded successfully!")
        model_error = None
        return pipeline
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(traceback.format_exc())
        model_error = str(e)
        return None
    finally:
        model_loading = False


# ============================================================
# Pydantic Models for OpenAI-compatible API
# ============================================================

class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.0


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="PaddleOCR-VL OpenAI Compatible API")
logger.info("✅ FastAPI app created")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "PaddleOCR-VL API", "status": "running"}


@app.get("/health")
@app.get("/ping")
async def health_check():
    """Health check endpoint for RunPod"""
    global pipeline, model_loading, model_error
    
    # If model is loading, return 204 (initializing)
    if model_loading:
        return JSONResponse(status_code=204, content={"status": "initializing"})
    
    # If model has error, still return healthy but indicate error
    if model_error:
        return {"status": "healthy", "model_loaded": False, "error": model_error}
    
    # If model not loaded, try to load in background
    if pipeline is None:
        import asyncio
        asyncio.create_task(asyncio.to_thread(load_model))
        return {"status": "healthy", "model_loaded": False, "loading": True}
    
    return {"status": "healthy", "model_loaded": True}


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
    """OpenAI-compatible chat completions endpoint."""
    import time
    import uuid
    
    logger.info(f"Received chat completion request for model: {request.model}")
    
    try:
        # Extract image from messages
        image_data = None
        prompt_text = ""
        
        for message in request.messages:
            content = message.content
            logger.info(f"Processing message role={message.role}, content_type={type(content)}")
            
            if isinstance(content, list):
                for i, item in enumerate(content):
                    logger.info(f"  Item {i}: type={type(item)}")
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type == "image_url":
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                            logger.info(f"  Found image_url, url_prefix={url[:50] if url else 'empty'}...")
                            if url.startswith("data:image"):
                                # Extract base64 from data URL
                                parts = url.split(",", 1)
                                if len(parts) == 2:
                                    image_data = parts[1]
                                    logger.info(f"  Extracted base64 data, length={len(image_data)}")
                                else:
                                    logger.warning("  Invalid data URL format")
                        elif item_type == "text":
                            prompt_text = item.get("text", "")
            elif isinstance(content, str):
                prompt_text = content
        
        if not image_data:
            logger.warning("No image provided in request")
            return JSONResponse(
                status_code=400, 
                content={"error": {"message": "No image provided", "type": "invalid_request_error"}}
            )
        
        logger.info(f"Image base64 length: {len(image_data)}")
        
        # Load model if not loaded
        model = load_model()
        if model is None:
            logger.error("Model not available")
            return JSONResponse(
                status_code=503,
                content={"error": {"message": f"Model not available: {model_error}", "type": "service_unavailable"}}
            )
        
        # Decode base64 to bytes
        try:
            # Clean base64 string (remove whitespace, newlines)
            image_data = image_data.strip().replace('\n', '').replace('\r', '').replace(' ', '')
            image_bytes = base64.b64decode(image_data)
            logger.info(f"Decoded image bytes: {len(image_bytes)}")
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Invalid base64 image: {e}", "type": "invalid_request_error"}}
            )
        
        # Validate it's a real image by checking magic bytes
        if len(image_bytes) < 8:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Image data too short", "type": "invalid_request_error"}}
            )
        
        # Detect image format from magic bytes
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            suffix = ".png"
        elif image_bytes[:2] == b'\xff\xd8':
            suffix = ".jpg"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            suffix = ".gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            suffix = ".webp"
        else:
            suffix = ".png"  # Default to PNG
            logger.warning(f"Unknown image format, magic bytes: {image_bytes[:8].hex()}")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name
        
        logger.info(f"Saved image to: {temp_path} (size: {len(image_bytes)} bytes, format: {suffix})")
        
        # Verify file was written correctly
        if not os.path.exists(temp_path):
            return JSONResponse(
                status_code=500,
                content={"error": {"message": "Failed to save temp image", "type": "internal_error"}}
            )
        
        file_size = os.path.getsize(temp_path)
        logger.info(f"Temp file size on disk: {file_size} bytes")
        
        try:
            # Run OCR
            start_time = time.time()
            results = list(model.predict(temp_path))
            elapsed = time.time() - start_time
            logger.info(f"OCR completed in {elapsed:.2f}s")
            
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
                
                if isinstance(page_data, dict):
                    if 'text' in page_data:
                        output_text += page_data['text'] + "\n"
                    elif 'rec_texts' in page_data:
                        output_text += "\n".join(page_data['rec_texts']) + "\n"
                    else:
                        output_text += json.dumps(page_data) + "\n"
                else:
                    output_text += str(page_data) + "\n"
            
            logger.info(f"Extracted {len(output_text)} characters")
            
            # Build response
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output_text.strip()
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": len(output_text.split()),
                    "total_tokens": 100 + len(output_text.split())
                }
            }
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}}
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Get port from environment (RunPod sets this)
    port = int(os.environ.get("PORT", 8008))
    
    logger.info(f"Starting server on port {port}")
    logger.info(f"Environment PORT={os.environ.get('PORT', 'not set')}")
    logger.info(f"Environment PORT_HEALTH={os.environ.get('PORT_HEALTH', 'not set')}")
    
    # Start loading model in background (don't block startup)
    import threading
    threading.Thread(target=load_model, daemon=True).start()
    
    # Run server - this must not block or crash
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
