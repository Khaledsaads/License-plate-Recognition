from fastapi import FastAPI, UploadFile, File, HTTPException
from plate_tracker import PlateTracker
from plate_visualizer import LicensePlateVisualizer
from pathlib import Path
import shutil
import os
from ultralytics import YOLO 
import uuid

app = FastAPI(title= 'License Plate recognition')

coco_model = YOLO(r'yolov8n.pt')
plate_model = YOLO(r'runs\detect\train-3\weights\best.pt')
plate_text_model = YOLO(r'Arabic_chars&numbers_yolov8n&yolov8s_new_data4-20260628T175839Z-3-001\Arabic_chars&numbers_yolov8n&yolov8s_new_data4\train-2\weights\best.pt')
plate_track = PlateTracker(coco_model, plate_model, plate_text_model)
plate_visualize = LicensePlateVisualizer()

VIDEOS = Path('videos_source')
Path.mkdir(VIDEOS, exist_ok= True)
OUPTUTS = Path('videos_dist')
Path.mkdir(OUPTUTS, exist_ok=True)
CSVS = Path('csv_results')
Path.mkdir(CSVS, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

@app.post('/track')
async def record_plates(file: UploadFile= File(...)):
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code= 400,
            detail= "Unsupported file extension."
        )
    job_id = str(uuid.uuid4())
    video_path = f'{VIDEOS}/{job_id}{extension}'
    csv_path = f'{CSVS}/{job_id}.csv'
    with open(video_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    plate_track.process_video(video_path, csv_path)

    return {
        'Job ID': job_id
    }

@app.post('/annotate/{job_id}')
async def annotate(job_id: str):
    csv_path = f'{CSVS}/{job_id}.csv'
    if not Path(csv_path).exists():
        raise HTTPException(
            status_code= 400,
            detail= "CSV File not existed"
        )
    video_files = list(Path('videos_source').glob(f'{job_id}.*'))
    if not video_files:
        raise HTTPException(
            status_code = 400,
            detail= 'Video file not existed'
        )
    video_file = video_files[0].name
    video_path = f"{VIDEOS}/{video_file} "
    output_path = Path(f'{OUPTUTS}/{video_file}')
    plate_visualize.process_video(video_path, output_path, csv_path)
    return {
        'video':video_file
    }