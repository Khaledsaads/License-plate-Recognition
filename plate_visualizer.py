import cv2
import pandas as pd 
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from bidi.algorithm import get_display


class LicensePlateVisualizer:
    
    def __init__(self, font_path= "C:\\Windows\\Fonts\\arial.ttf", font_size = 40):
        self.font_path = font_path
        self.font_size = font_size
        try:
            self.font = ImageFont.truetype(self.font_path, self.font_size)
        except IOError:
            raise IOError(f"Could not load font at {font_path}")
    
  
    def _prepare_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f'Could not find video: {input_path}')
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        return cap, out, total_frames


    def _extract_best_plates(self, cvs_path):
        results = pd.read_csv(cvs_path)
        license_plates = {}
        for car_id in np.unique(results['car_id']):
            max_ = np.max(results[results['car_id']== car_id]['license_number_score'])
            license_plates[car_id] = {'license_plate_number': results[ (results['car_id']== car_id)&\
                (results['license_number_score']== max_ )]['license_plate_number'].iloc[0]}
        
        return results, license_plates

    def _draw_separated_arabic_text(self, frame, text, position, font_size, color):
        bidi_text = get_display(text)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_frame)
        draw.text(position, bidi_text, font= self.font, fill= color)
        return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

    def process_video(self, input_path, output_path, csv_path):

        cap, out, total_frames = self._prepare_video(input_path, output_path)
        result, license_plates = self._extract_best_plates(csv_path)
        frame_nmr = -1
        ret = True
        while ret:
            frame_nmr+=1
            ret, frame = cap.read()
            df_ = result[result['frame_nmr']== frame_nmr]
            for row_idx in range(len(df_)):
                # extract car coordinates
                xcar1, ycar1, xcar2, ycar2 = json.loads(df_.iloc[row_idx]['car_bbox'])
                xcar1, ycar1, xcar2, ycar2 = int(xcar1), int(ycar1), int(xcar2), int(ycar2)
                # draw rectangle around car
                cv2.rectangle(frame, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 20)
                # extract plate coordinate
                x1, y1, x2, y2 =  json.loads(df_.iloc[row_idx]['license_plate_bbox'])
                x1, y1, x2, y2, = int(x1), int(y1), int(x2), int(y2)
                # draw box around license_plate
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 6)
                # get car id
                car_id = df_.iloc[row_idx]['car_id']
                # get highest score text's car id 
                try:
                    # 
                    license_plate_number = license_plates[df_.iloc[row_idx]['car_id']]['license_plate_number']
                    text_size = cv2.getTextSize(license_plate_number, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
                    tbx1, tbx2 = xcar1+30, xcar1+text_size[0]
                    tby1, tby2 = ycar1-30, ycar1- text_size[1]-50
                    cv2.rectangle(frame,(tbx1, tby1), (tbx2, tby2), (255, 255, 255), -1)
                    tx1, ty1 = xcar1+35, tby1-45
                    frame = self._draw_separated_arabic_text(frame, 
                                                        license_plate_number, 
                                                        (tx1, ty1), 
                                                        self.font_size, 
                                                        color=(0, 0, 250)
                                                    )
                except Exception as e:
                    raise ValueError(f'Frame= {frame_nmr} cause error: {e} ')
            
            out.write(frame)
            if frame_nmr%100 ==0 :
                print(f'Processed frames: -{frame_nmr}/{total_frames}-')
        cap.release()
        out.release()