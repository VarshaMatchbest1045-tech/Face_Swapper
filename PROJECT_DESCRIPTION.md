# FaceSwapper Project Documentation

## Project Overview
FaceSwapper is a FastAPI-based application designed to perform face swapping on images and videos. It utilizes deep learning models (specifically InsightFace) to detect and replace faces in a target media file with a source face. The project is structured as a modular Python application with a clear separation between the API layer, core processing logic, and specific frame processors.

## Directory Structure

### Root Directory (`/FaceSwapper`)
- **`run.py`**: The entry point for the application. It initializes a FastAPI server, defines the `/swap` endpoint, handling file uploads, and orchestrates the face-swapping process. It uses a thread lock to ensure sequential processing of requests.
- **`requirements.txt`**: Lists all Python dependencies required to run the project, including libraries like `fastapi`, `opencv-python`, `insightface`, `onnxruntime`, and `tensorflow`.
- **`README.md`**: General project documentation (often contains installation and usage instructions).
- **`CONTRIBUTING.md`**: Guidelines for potential contributors.
- **`.gitignore`**: Specifies files and directories that Git should ignore (e.g., temporary files, environments).
- **`.flake8` & `mypy.ini`**: Configuration files for code linting and type checking.

### Data Directories
- **`models/`**: Stores the heavy deep learning models required for operation (e.g., `inswapper_128.onnx`).
- **`uploads/`**: Temporary storage for source and target files uploaded by users.
- **`outputs/`**: Storage for the processed results (swapped images/videos).

### `faceswapper_core/` Package
This is the core package containing the application logic.

- **`__init__.py`**: Makes the directory a Python package.
- **`core.py`**: The "brain" of the application. It handles:
    -   Argument parsing (if run as CLI).
    -   Resource management (limiting memory/threads).
    -   Pre-checks (sanity checks for ffmpeg, python version).
    -   The main orchestration pipeline (`start()` function) which manages frame extraction, processing, and video re-assembly.
- **`globals.py`**: Contains global variables used across the application to maintain state (e.g., file paths, settings like `keep_fps`, `execution_providers`).
- **`utilities.py`**: a comprehensive collection of helper functions for:
    -   File analysis (detecting image/video types).
    -   Media processing (extracting frames using ffmpeg, creating videos).
    -   File system management (cleaning temp files).
- **`predictor.py`**: Contains logic for predicting/detecting specific content, likely including NSFW detection to prevent misuse.
- **`typing.py`**: Defines custom type hints for cleaner code (e.g., `Face`, `Frame`).
- **`face_analyser.py`**: Wrapper around face detection models to find and analyze faces in frames.
- **`face_reference.py`**: Manages the reference face (the source face) used for swapping.

### `faceswapper_core/processors/`
A modular directory for different types of processors.

#### `faceswapper_core/processors/frame/`
Contains processors that operate on individual frames.

- **`core.py`**: Logic to load and manage available frame processors.
- **`face_swapper.py`**: The core face-swapping implementation.
    -   Loads the `inswapper_128.onnx` model.
    -   Implements `pre_check` to ensure models are downloaded.
    -   Implements `process_frame` to detect faces in a frame and replace them with the source face.
- **`face_enhancer.py`**: (Likely) Uses GFPGAN or similar models to upscale and restore details in the swapped face to improve quality.

## Key Workflows

### 1. The Startup Flow (`run.py`)
1.  The FastAPI app acts as the interface.
2.  On startup, `init_app()` calls `faceswapper_core.core.pre_check()` to verify the environment.
3.  It pre-loads processors to minimize latency for the first request.

### 2. The Swap Request Flow
1.  **Upload**: User sends a POST request to `/swap` with a Source Image and Target (Image/Video).
2.  **Storage**: Files are saved with unique IDs in `uploads/`.
3.  **Configuration**: Global settings in `faceswapper_core.globals` are updated with the request parameters (paths, settings).
4.  **Processing**: `faceswapper_core.core.start()` is invoked:
    -   **Video**: Frames are extracted to a temporary folder -> Each frame is processed by `face_swapper` -> Frames are combined back into a video -> Audio is restored.
    -   **Image**: The single image is processed directly.
5.  **Cleanup**: Temporary frames are deleted.
6.  **Response**: The result from `outputs/` is returned to the user.

## Dependencies & requirements
The project relies heavily on **ONNX Runtime** for efficient model inference, supporting both CPU and CUDA (NVIDIA GPU) execution providers. **FFmpeg** is a system-level dependency required for all video operations.
