import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict, deque

# Load YOLO model (ensure best.pt is in the same directory)
model = YOLO('best.pt') 

# Initialize EasyOCR Reader (use gpu=True if you have an NVIDIA GPU)
reader = easyocr.Reader(['en'], gpu=False) 

# Corrected Regex: Matches 2 Letters, 2 Numbers, 3 Letters (e.g., AB12CDE)
plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$')

# Mapping for common OCR errors
dict_char_to_int = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'}
dict_int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}

def correct_format(text):
    """Aligns OCR output with Alpha-Alpha-Num-Num-Alpha-Alpha-Alpha format""" 
    if len(text) != 7: return None
    res = ""
    for i in range(7):
        char = text[i]
        if i in [0, 1, 4, 5, 6]: # Positions that MUST be Alphabets
            res += dict_int_to_char.get(char, char) if char.isdigit() else char
        else: # Positions that MUST be Numbers
            res += dict_char_to_int.get(char, char) if not char.isdigit() else char
    return res

def recognize_plate(plate_crop):
    """Pre-processes the crop and performs OCR""" 
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    # Thresholding to improve text contrast
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Resize to improve readability for small plates
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # EasyOCR returns a list of tuples: (bbox, text, confidence)
    results = reader.readtext(resized)
    for (_, text, conf) in results:
        clean_text = text.upper().replace(" ", "")
        candidate = correct_format(clean_text)
        if candidate and plate_pattern.match(candidate):
            return candidate
    return None

# Buffer stores predictions per vehicle ID for majority voting
stabilization_buffer = defaultdict(lambda: deque(maxlen=15))

cap = cv2.VideoCapture('vehicle_video.mp4') 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Use .track() to persist IDs across frames for better stabilization
    results = model.track(frame, persist=True, verbose=False)
    
    for result in results:
        if result.boxes is None or result.boxes.id is None:
            continue
            
        # Extract boxes, confidence, and track IDs
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, ids, confs):
            if conf < 0.3: continue
            
            x1, y1, x2, y2 = box
            # Ensure crop coordinates are within frame boundaries
            plate_crop = frame[max(0, y1):min(frame.shape[0], y2), 
                               max(0, x1):min(frame.shape[1], x2)]
            
            if plate_crop.size == 0: continue
            
            raw_text = recognize_plate(plate_crop)
            if raw_text:
                stabilization_buffer[track_id].append(raw_text)
            
            # Majority Voting: Pick the most frequent prediction for this specific vehicle
            if stabilization_buffer[track_id]:
                votes = list(stabilization_buffer[track_id])
                stable_text = max(set(votes), key=votes.count)
                
                # Draw bounding box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}: {stable_text}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('License Plate Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 

cap.release()
cv2.destroyAllWindows()
