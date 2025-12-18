import runpod
import base64
import tempfile
import json
import os

pipeline = None

def load_model():
    """Load PaddleOCR-VL model once, reuse for all requests."""
    global pipeline
    if pipeline is None:
        from paddlex import create_pipeline
        print("Loading PaddleOCR-VL model...")
        pipeline = create_pipeline("PaddleOCR-VL")
        print("Model loaded successfully!")
    return pipeline

def handler(event):
    """
    RunPod serverless handler for PaddleOCR-VL.

    Input:
        {
            "input": {
                "image_base64": "base64_encoded_image_string"
            }
        }

    Output:
        {
            "status": "success",
            "result": { ... extracted OCR data ... }
        }
    """
    try:
        # Get input
        input_data = event.get("input", {})
        image_base64 = input_data.get("image_base64")

        if not image_base64:
            return {"status": "error", "error": "No image_base64 provided"}

        # Decode image and save to temp file
        image_bytes = base64.b64decode(image_base64)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        try:
            # Load model and run inference
            model = load_model()
            results = list(model.predict(temp_path))

            # Process results into JSON-serializable format
            output = {
                "pages": []
            }

            for page_idx, page_result in enumerate(results):
                page_data = {
                    "page_number": page_idx + 1,
                    "blocks": []
                }

                # Extract blocks/regions from result
                if hasattr(page_result, 'json'):
                    page_data["raw"] = page_result.json
                elif hasattr(page_result, 'to_dict'):
                    page_data["raw"] = page_result.to_dict()
                elif isinstance(page_result, dict):
                    page_data["raw"] = page_result
                else:
                    page_data["raw"] = str(page_result)

                output["pages"].append(page_data)

            return {"status": "success", "result": output}

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
