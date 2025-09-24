from PIL import Image
import numpy as np
import cv2

image_path = '../../dataset/raw_dataset/skin_tone_testset/tenz.jpeg'
img = Image.open(image_path)

image_array = np.array(img)

print(f"image array shape: {image_array.shape}")
print(f"image array data type: {image_array.dtype}")

face_cascade = cv2.CascadeClassifier('../../open_cv_models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../../open_cv_models/haarcascade_eye.xml')

gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

print(gray_image)

#for face detection
faces = face_cascade.detectMultiScale(gray_image, 1.1, 4, minSize=(40, 40))

for (x, y, w, h) in faces:
    #for eye detection
    eye_gray_roi = gray_image[y:y+h, x:x+w]
    eye_image_roi = image_array[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(eye_gray_roi)
    print(f"eyes: {eyes}")
    if len(eyes) != 0:
        cv2.rectangle(image_array, (x, y), (x + w, y + h), (255, 0, 0), 2)


    for(ex, ey, ew, eh) in eyes:
        cv2.rectangle(eye_image_roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


cv2.imshow('detected faces', image_array)
cv2.waitKey(0)

cv2.destroyAllWindows()
