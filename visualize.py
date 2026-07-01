import cv2
import pandas as pd 
import numpy as np
import json

vidoe_path = r'plate_test1.mp4'
cap = cv2.VideoCapture(vidoe_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))
cvs_path = './test.csv'
results = pd.read_csv(cvs_path)
license_plates = {}
for car_id in np.unique(results['car_id']):
    max_ = np.max(results[results['car_id']== car_id]['license_number_score'])
    license_plates[car_id] = {'license_crop': None,
    'license_plate_number': results[ (results['car_id']== car_id)& (results['license_number_score']== max_ )]['license_plate_number'].iloc[0]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[ (results['car_id']== car_id)& (results['license_number_score']== max_ )]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    # x1, y1, x2, y2 =json.load(results[ (results['car_id']== car_id)& (results['license_number_score']== max_ )]['license_plate_bbox'].iloc[0])
    # license_crop = frame[int(y1): int(y2), int(x1): int(x2),:]
    # license_crop = cv2.resize(license_crop, (int((x2-x1)* 400/ (y2-y1)), 400))
    # license_plates[car_id]['license_crop']= license_crop

cap.set(cv2.CAP_PROP_FRAME_COUNT, 0)
class Visualizer:
    def draw(self, video_source,output_source, result, license_plates):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f'Could not open video: {video_source}')
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 12)
                # get car id
                car_id = df_.iloc[row_idx]['car_id']
                # get highest score text's car id 
                license_plate_number = license_plates[df_.iloc[row_idx]['car_id']]['license_plate_number']
                text_size = cv2.getTextSize(license_plate_number, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 2)[0]
                tbx1, tbx2 = xcar1+5, xcar1+text_size[0]+20
                tby1, tby2 = ycar1-5, ycar1- text_size[1]-10
                cv2.rectangle(frame,(tbx1, tby1), (tbx2, tby2), (255, 255, 255), -1)
                tx1, ty1 = xcar1+10, ycar1-10
                cv2.putText(frame, license_plate_number, (tx1, ty1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            
            out.write(frame)
            if frame_nmr%100 ==0 :
                print(frame_nmr)
        cap.release()
        out.release()

visualize = Visualizer()
visualize.draw(r'plate_test1.mp4',out, results, license_plates)