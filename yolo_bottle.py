from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture("bottles.mp4")
unique_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform tracking - Class 39 is 'bottle'
    results = model.track(frame, classes=[39], persist=True, verbose=False)
    
    
    annotated_frame = results[0].plot()
    
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        
        ids = results[0].boxes.id.int().tolist()
        
        for oid in ids:
            unique_ids.add(oid)
            
    # Draw the cumulative count on the screen
    cv2.putText(annotated_frame, f"Total Bottles: {len(unique_ids)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Object Tracking", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    