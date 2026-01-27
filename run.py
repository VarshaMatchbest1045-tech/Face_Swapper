import os
import shutil
import glob
import uuid
import threading
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn

import faceswapper_core.globals
import faceswapper_core.core
import faceswapper_core.metadata
from faceswapper_core.processors.frame.core import get_frame_processors_modules

app = FastAPI(title="FaceSwapper API", version=faceswapper_core.metadata.version)

# Global lock to prevent race conditions since roop uses global variables
process_lock = threading.Lock()

def init_app():
    """Initialize roop (checks and resource limits)."""
    faceswapper_core.globals.headless = True
    if not faceswapper_core.core.pre_check():
        raise Exception("Roop pre-check failed. Check ffmpeg and python version.")
    faceswapper_core.core.limit_resources()
    # Pre-load processors to avoid delay on first request
    # This might take time on startup
    print("Pre-loading processors...")
    for processor in faceswapper_core.globals.execution_providers:
        pass 

@app.on_event("startup")
async def startup_event():
    init_app()

@app.get("/")
def read_root():
    return {"message": "Welcome to FaceSwapper API", "version": faceswapper_core.metadata.version}

@app.post("/swap")
def swap_faces(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    face_enhancer: bool = Form(False),
    keep_fps: bool = Form(True),
    skip_audio: bool = Form(False),
    many_faces: bool = Form(False),
):
    """
    Swap faces from source image to target image/video.
    """
    # Create unique session ID for file paths to avoid collisions (though we lock anyway)
    session_id = str(uuid.uuid4())
    upload_dir = "uploads"
    output_dir = "outputs"
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Define paths
    source_ext = source.filename.split('.')[-1] if '.' in source.filename else "jpg"
    target_ext = target.filename.split('.')[-1] if '.' in target.filename else "mp4"
    
    # Simple validation based on extension
    if target_ext.lower() in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
        output_ext = target_ext
    else:
        output_ext = "mp4"

    source_path = os.path.join(upload_dir, f"{session_id}_source.{source_ext}")
    target_path = os.path.join(upload_dir, f"{session_id}_target.{target_ext}")
    output_path = os.path.join(output_dir, f"output_{session_id}.{output_ext}")

    try:
        # Save files
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source.file, f)
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target.file, f)

        # Acquire lock and process
        # This block is synchronous and will block the thread, effectively queuing requests
        with process_lock:
            # Configure globals
            faceswapper_core.globals.source_path = os.path.abspath(source_path)
            faceswapper_core.globals.target_path = os.path.abspath(target_path)
            faceswapper_core.globals.output_path = os.path.abspath(output_path)
            faceswapper_core.globals.headless = True
            
            # Frame processors
            processors = ['face_swapper']
            if face_enhancer:
                processors.append('face_enhancer')
            faceswapper_core.globals.frame_processors = processors
            
            faceswapper_core.globals.keep_fps = keep_fps
            faceswapper_core.globals.keep_frames = False # Always clean up temp frames
            faceswapper_core.globals.skip_audio = skip_audio
            faceswapper_core.globals.many_faces = many_faces
            faceswapper_core.globals.reference_face_position = 0
            faceswapper_core.globals.reference_frame_number = 0
            faceswapper_core.globals.similar_face_distance = 0.85
            faceswapper_core.globals.temp_frame_format = 'png'
            faceswapper_core.globals.temp_frame_quality = 100
            faceswapper_core.globals.output_video_encoder = 'libx264'
            faceswapper_core.globals.output_video_quality = 35
            faceswapper_core.globals.max_memory = None # default
            # Use default providers (CPU/CUDA) determined by faceswapper_core
            faceswapper_core.globals.execution_providers = faceswapper_core.core.decode_execution_providers(['cpu', 'cuda']) 
            faceswapper_core.globals.execution_threads = faceswapper_core.core.suggest_execution_threads()

            print(f"Starting processing for session {session_id}...")
            
            # Run the core logic
            # faceswapper_core.core.start() handles everything including creating temp frames, processing, and cleaning them
            faceswapper_core.core.start()
            
            # faceswapper_core.core.destroy() is usually called to exit, but we modified it to just clean. 
            # We should call clean_temp explicitly to be safe, although start() does it at the end.
            faceswapper_core.core.clean_temp(faceswapper_core.globals.target_path)

            print(f"Processing finished for session {session_id}.")

        # Check if output exists
        if os.path.exists(output_path):
            return FileResponse(output_path, media_type="application/octet-stream", filename=f"swapped_{target.filename}")
        else:
            raise HTTPException(status_code=500, detail="Processing failed, no output generated. Check server logs.")

    except Exception as e:
        print(f"Error during processing: {e}")
        # Cleanup on error
        if os.path.exists(source_path):
            os.remove(source_path)
        if os.path.exists(target_path):
            os.remove(target_path)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup uploaded files (source and target)
        # Note: We might want to keep the output file for a bit or rely on OS to clean tmp
        # Here we attempt to clean source/target but getting file handles released is tricky on Windows sometimes.
        try:
            if os.path.exists(source_path):
                os.remove(source_path)
            if os.path.exists(target_path):
                os.remove(target_path)
        except Exception as cleanup_error:
            print(f"Warning: Failed to cleanup files: {cleanup_error}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
