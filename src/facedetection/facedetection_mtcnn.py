from mtcnn import MTCNN
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the video")
    exit()

while cap.isOpened():
    # ret -> boolean
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        
        image = Image.fromarray(rgb_frame)

        print(f"Image successfully loaded: {image.format}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(device)   

        mtcnn_model = MTCNN(device=device) 
        detection_result = mtcnn_model.detect_faces(image)

        plt.show(plot(image, detection_result))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else: 
        break

cap.release()
cv2.destroyAllWindows()
