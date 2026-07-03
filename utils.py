from ultralytics import  YOLO
import json

yolo_to_arabic = {
    "0": "٠",
    "1": "١",
    "2": "٢",
    "3": "٣",
    "4": "٤",
    "5": "٥",
    "6": "٦",
    "7": "٧",
    "8": "٨",
    "9": "٩",

    "alif": "أ",
    "alif_maqsura": "ى",
    "baa": "ب",
    "taa": "ت",
    "thaa": "ث",
    "jeem": "ج",
    "7aa": "ح",
    "khaa": "خ",
    "daal": "د",
    "zaal": "ذ",
    "raa": "ر",
    "zay": "ز",
    "seen": "س",
    "sheen": "ش",
    "saad": "ص",
    "daad": "ض",
    "Taa": "ط",
    "Thaa": "ظ",
    "ain": "ع",
    "ghayn": "غ",
    "faa": "ف",
    "qaaf": "ق",
    "kaaf": "ك",
    "laam": "ل",
    "meem": "م",
    "noon": "ن",
    "haa": "هـ",
    "waw": "و",
    "yaa": "ي"
}

def get_text(model, crop_plate= None):
    extracted_boxes = []
    text_result = model(crop_plate, verbose = False)[0]
    for i, box in enumerate(text_result.boxes):
        x1, y1, x2, y2= box.xyxy[0].cpu().tolist()
        score = box.conf.item()
        cls_id = int(box.cls.item())
        class_name = text_result.names[cls_id]
        extracted_boxes.append({
            'x2': x2,
            'score': score,
            'class': class_name
        })
    extracted_boxes = sorted(extracted_boxes, key = lambda x : x['x2'], reverse=True)
    final_conf = 0.0
    plate_text = ''
    for box in extracted_boxes:
        ch= box['class']
        plate_text += yolo_to_arabic[ch]
        plate_text+= ' '
        final_conf+= box['score']
    if not len(extracted_boxes) ==0:
        final_conf/=len(extracted_boxes)
    return plate_text, final_conf




def write_csv(results, output_path):  
    with open(output_path, 'w', encoding= 'utf-8') as f:
        # Header
        f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number_score,license_plate_number\n')

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                data = results[frame_nmr][car_id]
                
                if 'car' in data and 'license_plate' in data and 'license_plate_number' in data['license_plate']\
                    and data['license_plate']['license_plate_number']:
                    #  encode lists to JSON strings wrapped in double quotes
                    car_bbox = json.dumps(data['car']['bbox'])
                    lp_bbox = json.dumps(data['license_plate']['bbox'])
                    
                    f.write(
                        f'{frame_nmr},{car_id},"{car_bbox}","{lp_bbox}",'
                        f'{data["license_plate"]["bbox_score"]},'
                        f'{data["license_plate"]["text_score"]},'
                        f'{data["license_plate"]["license_plate_number"]}\n'
                    )

def Get_car(license_plate, vehicle_ids):
    x1, y1, x2, y2, plate_id, score = license_plate
    foundit = False
    car_idx = None
    for i in range(len(vehicle_ids)):
        xcar1, ycar1, xcar2, ycar2, vehicle_id = vehicle_ids[i]
        if (x1 > xcar1 and y1> ycar1) and (x2< xcar2 and y2< ycar2):
            foundit = True
            car_idx = i
            break
    if foundit:
        return vehicle_ids[i]
    else:
        return -1 , -1, -1, -1, -1