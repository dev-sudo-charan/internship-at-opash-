import cv2

capture = cv2.VideoCapture('Raw_video_of_a_home_burglary_caught_on_camera_1080P.mp4')
frames = []
gap = 5
count = 0 

while True:
    ret, frame = capture.read()

    if not ret:
        break

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

    
    if len(frames) > gap + 1:
        frames.pop(0)

    if len(frames) > gap:
       
        diff = cv2.absdiff(frames[0], frames[-1])
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for c in contours:
            if cv2.contourArea(c) < 1000:
                continue
            
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if motion_detected:
            cv2.putText(frame, "MOTION DETECTED!!!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imwrite(f"motion_frame_{count}.jpg", frame)

       
        cv2.imshow("Motion Detection", frame)
        count += 1

    
    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()