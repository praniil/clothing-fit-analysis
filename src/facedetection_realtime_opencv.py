import cv2
import time
import os
import test_finetuned_resnet18 as test

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

face_cascade = cv2.CascadeClassifier('../open_cv_models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../open_cv_models/haarcascade_eye.xml')

screenshot_count = 0;

folder_name = "../opencv_screenshots"
os.makedirs(folder_name, exist_ok=True)

dir = os.listdir(folder_name)
image_path_array = []

for item in dir:
    item_path = os.path.join(folder_name, item)
    if os.path.isfile(item_path):
        os.remove(item_path)
    
print(f"Content of {folder_name} cleared")

while 1:
    ret, frame = cap.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5, minSize=(10, 10))

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray_frame[y:y+h, x:x+w]
        face_crop = frame[y:y+h, x:x+w]

        # Detects eyes of different sizes in the input image
        # eyes = eye_cascade.detectMultiScale(roi_gray, minSize=(5, 5)) 

        # #To draw a rectangle in eyes
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(face_crop,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
        
    # Display an image in a window
    cv2.imshow('img', frame)

    current_timestamp_time = time.time() * 1000
    
    if screenshot_count < 5 and len(faces) > 0:
        print(current_timestamp_time)
        cv2.imwrite(f"{folder_name}/{current_timestamp_time}.jpg", face_crop)
        screenshot_count+=1
        image_path_array.append(f"{folder_name}/{current_timestamp_time}.jpg")

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

for i in range(len(image_path_array)):
    result = test.predict_skin_tone(image_path_array[i])
    print(f"Predicted Skin Tone for {image_path_array[i]} is: {result}")

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()


