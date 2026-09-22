from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import shutil
import os

from video_processor import process_video


app = FastAPI()


os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/upload")
def upload_video(file: UploadFile):

    input_path = f"uploads/{file.filename}"
    output_path = "outputs/annotated.mp4"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process_video(
        input_path,
        output_path
    )

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="annotated.mp4"
    )