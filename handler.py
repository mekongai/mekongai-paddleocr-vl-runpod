"""
RunPod Serverless Handler for PaddleOCR-VL
Returns both OCR text and cropped images (tables, figures, etc.)

Uses PaddleX pipeline.predict() for full output with markdown_images.
"""
import runpod
import base64
import io
import os
import tempfile
import time
import traceback
from typing import Dict, Any, Optional

print("=" * 60)
print("PaddleOCR-VL RunPod Handler - Starting")
print("=" * 60)

# Global pipeline instance
pipeline = None
pipeline_loading = False
pipeline_error = None


def load_pipeline():
    """Load PaddleOCR-VL pipeline once"""
    global pipeline, pipeline_loading, pipeline_error
    
    if pipeline is not None:
        return pipeline
    
    if pipeline_loading:
        return None
    
    pipeline_loading = True
    
    try:
        print("🚀 Loading PaddleOCR-VL pipeline...")
        
        # Enable dynamic mode - CRITICAL
        import paddle
        paddle.disable_static()
        print("✅ Paddle dynamic mode enabled")
        
        # Create pipeline
        from paddlex import create_pipeline
        pipeline = create_pipeline("PaddleOCR-VL")
        print("✅ Pipeline loaded successfully!")
        
        pipeline_error = None
        return pipeline
        
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        traceback.print_exc()
        pipeline_error = str(e)
        return None
    finally:
        pipeline_loading = False


def pil_to_base64(img, format="PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def download_image(url: str) -> bytes:
    """Download image from URL"""
    import requests
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def process_image(image_data: bytes) -> Dict[str, Any]:
    """
    Process image with PaddleOCR-VL pipeline.
    
    Returns:
        {
            "text": "OCR text content",
            "markdown_images": {
                "image_name.png": "base64_data",
                ...
            }
        }
    """
    global pipeline
    
    # Ensure pipeline is loaded
    if pipeline is None:
        pipeline = load_pipeline()
        if pipeline is None:
            return {"error": f"Pipeline not available: {pipeline_error}"}
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_data)
        temp_path = f.name
    
    try:
        print(f"📸 Processing image: {len(image_data)} bytes")
        start_time = time.time()
        
        # Run OCR with pipeline
        results = list(pipeline.predict(temp_path))
        
        elapsed = time.time() - start_time
        print(f"✅ OCR completed in {elapsed:.2f}s")
        
        # Extract text and images from result
        text_parts = []
        images_b64 = {}
        
        for result in results:
            # Get markdown output
            if hasattr(result, "_to_markdown"):
                md_result = result._to_markdown()
                
                if isinstance(md_result, dict):
                    # Get text
                    if "markdown_texts" in md_result:
                        text_parts.append(md_result["markdown_texts"])
                    elif "text" in md_result:
                        text_parts.append(md_result["text"])
                    
                    # Get images
                    if "markdown_images" in md_result:
                        for img_name, img_obj in md_result["markdown_images"].items():
                            try:
                                # img_obj is PIL Image
                                images_b64[img_name] = pil_to_base64(img_obj)
                                print(f"  📷 Extracted image: {img_name}")
                            except Exception as e:
                                print(f"  ⚠️ Failed to encode image {img_name}: {e}")
                else:
                    # md_result is string
                    text_parts.append(str(md_result))
            
            # Fallback: try other attributes
            elif hasattr(result, "json"):
                data = result.json
                if "text" in data:
                    text_parts.append(data["text"])
            elif hasattr(result, "text"):
                text_parts.append(result.text)
        
        final_text = "\n".join(text_parts)
        
        print(f"📝 Extracted: {len(final_text)} chars, {len(images_b64)} images")
        
        return {
            "text": final_text,
            "markdown_images": images_b64
        }
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        traceback.print_exc()
        return {"error": str(e)}
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless Handler
    
    Input format:
    {
        "input": {
            "image": "base64_encoded_string",   // Option 1: base64
            "image_url": "https://...",         // Option 2: URL
            "task": "ocr"                       // Optional: ocr, table, formula, chart
        }
    }
    
    Output format:
    {
        "text": "OCR text content",
        "markdown_images": {
            "table_0.png": "base64...",
            "figure_0.png": "base64..."
        },
        "model": "PaddleOCR-VL",
        "task": "ocr"
    }
    """
    try:
        input_data = job.get("input", {})
        
        # Get image data
        image_b64 = input_data.get("image")
        image_url = input_data.get("image_url")
        task = input_data.get("task", "ocr")
        
        if not image_b64 and not image_url:
            return {"error": "Missing image. Provide 'image' (base64) or 'image_url' (URL)"}
        
        # Get image bytes
        if image_url:
            print(f"📥 Downloading from URL: {image_url[:50]}...")
            
            # Handle data URL (base64 embedded in URL)
            if image_url.startswith("data:image"):
                parts = image_url.split(",", 1)
                if len(parts) == 2:
                    image_data = base64.b64decode(parts[1])
                else:
                    return {"error": "Invalid data URL format"}
            else:
                image_data = download_image(image_url)
        else:
            print("📦 Decoding base64 image...")
            image_data = base64.b64decode(image_b64)
        
        # Process image
        result = process_image(image_data)
        
        if "error" in result:
            return result
        
        # Add metadata
        result["model"] = "PaddleOCR-VL"
        result["task"] = task
        
        return result
        
    except Exception as e:
        print(f"❌ Handler error: {e}")
        traceback.print_exc()
        return {"error": str(e)}


# ============== MAIN ==============
if __name__ == "__main__":
    # Pre-load pipeline when starting
    print("🔧 Pre-loading pipeline...")
    load_pipeline()
    
    # Start RunPod handler
    print("🎯 Starting RunPod Serverless handler...")
    runpod.serverless.start({"handler": handler})