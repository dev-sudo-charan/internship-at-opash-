from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")

image = cv2.imread("bicycle.jpg")

results = model(image)

labeled_image = results[0].plot()

cv2.imshow("labeled image", labeled_image)   

cv2.waitKey(0)
cv2.destroyAllWindows() 
