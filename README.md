# PaddleOCR-VL RunPod Serverless

Deploy PaddleOCR-VL as a serverless API on RunPod.

## Deployment

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint** → **Import from GitHub**
3. Connect this repository
4. Select GPU: **RTX 3090** or **RTX 4090** (24GB VRAM required)
5. Deploy

## Usage

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_base64": "BASE64_ENCODED_IMAGE"
    }
  }'
```

## Response

```json
{
  "status": "success",
  "result": {
    "pages": [
      {
        "page_number": 1,
        "raw": { ... extracted OCR data ... }
      }
    ]
  }
}
```
