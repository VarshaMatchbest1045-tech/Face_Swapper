# FaceSwapper Backend

FaceSwapper has been converted into a powerful backend API for swapping faces in images and videos. It uses FastAPI to serve endpoints that process media using deep learning models.

## Features

- **REST API**: Simple `/swap` endpoint to upload and process files.
- **Face Swapping**: High-quality face swapping using `inswapper_128`.
- **Face Enhancement**: Optional GFPGAN enhancement.
- **Support**: Handles Images and Videos.
- **Concurrency**: Thread-safe processing (requests are queued).

## Requirements

- **Python**: Version 3.9 or higher.
- **FFmpeg**: Must be available in system PATH.
- **CUDA** (Optional): For GPU acceleration (highly recommended).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/VarshaMatchbest1045-tech/FaceSwapper.git
    cd FaceSwapper
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Server

Start the API server using `api.py` or via `uvicorn` directly:

```bash
python run.py
# OR
uvicorn run:app --host 0.0.0.0 --port 8000
```

The server will start on `http://0.0.0.0:8000`.

## API Documentation

Once the server is running, you can access the interactive API docs at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Endpoint: `POST /swap`

Swaps the face from a source image into a target image or video.

**Parameters (Form Data):**

-   `source`: (File, Required) The source image containing the face you want to use.
-   `target`: (File, Required) The target image or video where the face will be swapped.
-   `face_enhancer`: (Boolean, Optional, default `False`) Enable face enhancement (GFPGAN).
-   `keep_fps`: (Boolean, Optional, default `True`) Keep original FPS for video targets.
-   `skip_audio`: (Boolean, Optional, default `False`) Remove audio from output video.
-   `many_faces`: (Boolean, Optional, default `False`) Swap every detected face, not just the reference one.

**Example using cURL:**

```bash
curl -X 'POST' \
  'http://localhost:8000/swap' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'source=@mysource.jpg;type=image/jpeg' \
  -F 'target=@mytarget.mp4;type=video/mp4' \
  -F 'face_enhancer=true' \
  --output result.mp4
```

## Models

The system will automatically download necessary models (`inswapper_128.onnx`, `GFPGAN`, etc.) on the first run. Ensure you have internet access or place them in the `models/` directory manually.


