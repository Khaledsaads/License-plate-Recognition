import streamlit as st, requests
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import uuid
st.set_page_config(
    page_title='🪪 License Plate Monitoring System',
    page_icon='🪪',
    layout='wide'
)

BASE_URL = "http://127.0.0.1:8000"

st.title("🪪 Welcome License Plate Monitoring System")
st.markdown("""
> **Note:** Uploaded videos are processed in full, which may take some time, especially for long videos.  
> Once the videos have been processed, you can submit your queries and get results much faster.  
> If you only need a specific part of a video, please upload that part instead of the full video to reduce processing time.  
> For the best experience, please upload all the videos you need at once and wait for the initial processing to finish.
""")

st.markdown("""
### 🛠️ Available Services

- 🎯 **Annotate a Video** — Detect and track license plates in the video.
- 🔎 **Search for a Plate by Time Interval** — Find a specific license plate within a selected time range.
- 📈 **View Plate Occurrences Over Time** — See when and how often each plate appears.
- 🔢 **Count Unique Plates** — Get the total number of distinct license plates detected.
- 📋 **View All Plate Occurrences** — Browse all detected license plate occurrences and their details.
""")

def time_to_frames(time_, fps=30):
    hour, minute, sec = time_
    frame = (3600*hour + 60*minute + sec)*fps
    return frame

def frame_to_time(frame_nums, fps=30):
    frame_nums /= fps
    num_hour = int(frame_nums/3600)
    num_mins = int((frame_nums- num_hour* 3600)/ 60)
    num_secs = int(frame_nums - num_hour* 3600 - num_mins * 60)
    return (num_hour, num_mins, num_secs)
    
def save_full_video(uploaded_file):
    FULL_VIDEOS = Path("full_videos")
    FULL_VIDEOS.mkdir(parents=True, exist_ok=True)
    full_video_id = str(uuid.uuid4())
    file_path = FULL_VIDEOS/f'{full_video_id}.mp4'
    with open(file_path, 'wb')as f:
        f.write(uploaded_file.getbuffer())
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.session_state['default_end_time'] = frame_to_time(total_frames, fps)
    st.session_state['current_session_id'] = full_video_id


uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file:
    save_full_video(uploaded_file)


if 'current_session_id' not in st.session_state:
    st.warning("Please upload the video")
    st.stop()


def time_to_start():
    st.markdown("##### Enter start time")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_hour = st.number_input('Hour', min_value=0, max_value=3, value=0, step = 1, width= 70, key='start_hour')
    with col2:
        start_minute = st.number_input('Minute', min_value=0, max_value=59, value=0, width=70, key = 'start_min')
    with col3:
        start_second = st.number_input('Second', min_value=0, max_value=59, value=0, width=70,key = 'start_sec')
    return (start_hour, start_minute, start_second)

def time_to_end():
    st.markdown("##### Enter end time")
    col4, col5, col6 = st.columns(3)
    with col4:
        end_hour = st.number_input('Hour', min_value=0, max_value=3, value=st.session_state['default_end_time'][0], step = 1, width= 70, key = 'end_hour')
    with col5:
        end_minute = st.number_input('Minute', min_value=0, max_value=59, value=st.session_state['default_end_time'][1], width=70, key = 'end_min')
    with col6:
        end_second = st.number_input('Second', min_value=0, max_value=59, value=st.session_state['default_end_time'][2], width=70, key = 'end_sec')
    return (end_hour, end_minute, end_second)



def cut_video(start_time, end_time, video_source):
    CUTTED_PATH = Path('cutted_videos')
    CUTTED_PATH.mkdir(parents=True, exist_ok=True)
    output_id = str(uuid.uuid4())
    output_source = Path(CUTTED_PATH/f'{output_id}.mp4')
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Can not access Source file: {video_source}")
    
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cutted_video = cv2.VideoWriter(output_source, fourcc, fps, (width, height))
    start_frame= time_to_frames(start_time, fps)
    end_frame= time_to_frames(end_time, fps)
    end_frame = min(end_frame, total_frms)
    if start_frame < total_frms:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        raise ValueError("Your starting time exceeded video's time limit")
    cnt_frames = start_frame -1
    ret= True 
    while ret and cnt_frames < end_frame:
        cnt_frames+=1
        ret, frame = cap.read()
        cutted_video.write(frame)
    cap.release()
    cutted_video.release()
    st.session_state['fps'] = fps
    return output_source

start_time = time_to_start()
end_time = time_to_end()
confirm_time = st.button('Confirm time', type = 'primary')
if confirm_time:
  st.session_state['time_confirmed'] = True
  st.session_state['start_time'] = start_time
  st.session_state['end_time'] = end_time
  video_source = Path(f'full_videos/{st.session_state["current_session_id"]}.mp4')
  with st.spinner('Cutting video segment...'):
    st.session_state['cutted_video_dist'] = cut_video(
        start_time, end_time, video_source
    )
  st.session_state['last_start_time'] = start_time
  st.session_state['last_end_time'] = end_time
  st.rerun()
        
if not st.session_state.get('time_confirmed', False):
    st.warning("please confirm time first")
    st.stop()
if not st.session_state.get('Processed', False):
    with st.spinner("Start processing video..."):
        with open(st.session_state['cutted_video_dist'], 'rb')as f:
            files = {
                'file': f
            }
            response = requests.post(f'{BASE_URL}/track', files=files)
        if response.status_code ==200:
            operation_id = response.json()['Job ID']
            st.session_state['operation_id'] = operation_id
            df = pd.read_csv(f'csv_results/{st.session_state['operation_id']}.csv')
            license_plates = {}
            for car_id in np.unique(df['car_id']):
                mx = np.max(df[df['car_id']== car_id]['license_number_score'])
                license_plate_number = df[
                (df['car_id'] == car_id)&
                (df['license_number_score'] ==mx)]['license_plate_number'].iloc[0]
                first_frame = np.min(df[df['car_id']==car_id]['frame_nmr'])
                last_frame = np.max(df[df['car_id']==car_id]['frame_nmr'])
                license_plates[license_plate_number]= (first_frame, last_frame)
            st.session_state['license_plates'] = license_plates
            st.session_state['Processed'] = True
            st.success("Video Processed Successfully.")
        else:
            st.error("Processing failed.")

if st.session_state.get('Processed', False) :
    choice = st.selectbox(
    'what do you want to do?',
    [
        'Annotate a video',
        'Search for a plate within a specific time interval',
        'View plate occurrences over time',
        'Count unique plates',
        'View all plate occurrences',
    ],
    index= None,
    placeholder="Select an option...",
    )
  
    if choice == 'View all plate occurrences':
        data = []
        video_start = time_to_frames(st.session_state['start_time'])
        video_end = time_to_frames(st.session_state['end_time'])
        for plate, times in st.session_state['license_plates'].items():
            start_time_car = frame_to_time(video_start +times[0], st.session_state['fps'])
            end_time_car = frame_to_time(video_end +times[1], st.session_state['fps'])
            data.append({
                "License Plate": plate,
                "Start Time": start_time_car,
                "End Time": end_time_car
            })
        st.table(data)
                        

