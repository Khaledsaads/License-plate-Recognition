from utils import *
import cv2
class PlateTracker:
    def __init__(self, coco_model, plate_model, plate_text_model, target_vehicles=[2, 3, 5, 7]):
        self.coco_model = coco_model
        self.plate_model = plate_model
        self.text_model = plate_text_model
        self.target_vehicles = target_vehicles

    def process_video(self, video_source, output_csv_path, tracker_config='bytetrack.yaml'):
        total_frames = None
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result = {}
        ret = True
        frame_num = -1
        try:
            while ret:
                frame_num+=1
                ret, frame = cap.read()
                if not ret:
                    break
                result[frame_num]= {}
                track_results = self.coco_model.track(frame, persist = True, tracker = tracker_config, verbose= False)[0]
                track_ids = []
                
                # track cars
                if track_results and track_results.boxes.id  is not None:
                    boxes = track_results.boxes.xyxy.detach().cpu().tolist()
                    ids = track_results.boxes.id.detach().cpu().tolist()
                    clss = track_results.boxes.cls.detach().cpu().tolist()
                    scores = track_results.boxes.conf.detach().cpu().tolist()
                    # if class in vehicles add it to IDs
                    for class_id, box, id_ in zip(clss, boxes, ids):
                        if int(class_id) in self.target_vehicles:
                            x1, y1, x2, y2 = box
                            track_ids.append([x1, y1, x2, y2, int(id_)])


                license_plates = self.plate_model(frame, verbose= False)[0]
                for license_plate in license_plates.boxes.data.detach().cpu().tolist():
                    x1, y1, x2, y2, plate_score, plate_id = license_plate
                    xcar1, ycar1, xcar2, ycar2, car_id = Get_car(license_plate, track_ids)
                    # if car existed 
                    if car_id !=-1:
                        crop_plate = frame[int(y1): int(y2), int(x1):int(x2), :]
                        crop_plate_gray = cv2.cvtColor(crop_plate, cv2.COLOR_BGR2GRAY)
                        crop_plate_input = cv2.cvtColor(crop_plate_gray, cv2.COLOR_GRAY2BGR)
                        license_plate_text, text_score = get_text(model= self.text_model, crop_plate = crop_plate_input)
                        
                        # add car and it's plates
                        result[frame_num][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': plate_score,
                            'text_score': text_score}
                        }
                if not frame_num%100:
                    print(f'Processed frames: -{frame_num}/{total_frames}-')
                    
            write_csv(result, output_csv_path)
        finally:
            # Ensures resources are freed even if an error occurs mid-loop
            cap.release()