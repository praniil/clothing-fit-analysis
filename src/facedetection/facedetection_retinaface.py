from retinaface import RetinaFace
import cv2
import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info
print("TensorFlow:", tf.__version__)
print("TF built with CUDA:", tf_build_info.build_info['cuda_version'])
print("TF built with cuDNN:", tf_build_info.build_info['cudnn_version'])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while 1:
    ret, frame = cap.read() 

    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]

    resp = RetinaFace.detect_faces(rgb_frame)

    if isinstance(resp, dict):
        for key in resp:
            identity = resp[key]

            facial_area = identity['facial_area']
            landmarks = identity['landmarks']
            score = identity['score']

    cv2.rectangle(frame, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (0, 255, 0), 2)

    if landmarks:
        for point in landmarks.values():
            x, y = point[:2]
            center = (int(x), int(y))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    cv2.imshow('RetinaFace Real-time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    