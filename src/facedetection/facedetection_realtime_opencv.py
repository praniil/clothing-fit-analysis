import cv2
import time
import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Python can now find modules in any subfolder of src.
from skin_tone_classifier.test_finetuned_resnet18 import predict_skin_tone


#capture the video frames
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# load the opencv models
face_cascade = cv2.CascadeClassifier('../../open_cv_models/haarcascade_frontalface_default.xml')

# global count variables
screenshot_count = 0;
total_screenshot_need = 15;
brown = 0
white = 0
black = 0

# maintain the opencv_screenshot folder -- delete the previous screenshots
folder_name = "../../opencv_screenshots"
os.makedirs(folder_name, exist_ok=True)

dir = os.listdir(folder_name)
image_path_array = []

for item in dir:
    item_path = os.path.join(folder_name, item)
    if os.path.isfile(item_path):
        os.remove(item_path)
    
print(f"Content of {folder_name} cleared")

# video capture loop
while 1:
    ret, frame = cap.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5, minSize=(10, 10))

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        face_crop = frame[y:y+h, x:x+w]
        
    # Display an image in a window
    cv2.imshow('img', frame)

    current_timestamp_time = time.time() * 1000
    
    if screenshot_count < total_screenshot_need and len(faces) > 0:
        cv2.imwrite(f"{folder_name}/{current_timestamp_time}.jpg", face_crop)
        image_path_array.append(f"{folder_name}/{current_timestamp_time}.jpg")
        result = predict_skin_tone(image_path_array[-1])

        if result == 'Brown':
            brown += 1
        elif result == 'White':
            white += 1
        else:
            black += 1

        if screenshot_count == total_screenshot_need - 1:
            if brown > white and brown > black:
                end_result = "Brown"
            elif white > brown and white > black:
                end_result = "White"
            else:
                end_result = "Black"

            print(f"Brown: {brown}, Black: {black}, White: {white}")

            print(f"Predicted Skin Tone: {end_result}")

        screenshot_count+=1

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()


