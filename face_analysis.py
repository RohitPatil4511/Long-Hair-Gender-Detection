import cv2
import numpy as np
import os

# ===============================
# Model Paths
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

faceProto = os.path.join(BASE_DIR, "models", "deploy.prototxt")
faceModel = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")

ageProto = os.path.join(BASE_DIR, "models", "age_deploy.prototxt")
ageModel = os.path.join(BASE_DIR, "models", "age_net.caffemodel")

genderProto = os.path.join(BASE_DIR, "models", "gender_deploy.prototxt")
genderModel = os.path.join(BASE_DIR, "models", "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

genderList = ['Male', 'Female']

# ===============================
# Load Models
# ===============================

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


# ===============================
# Face Detection Function
# ===============================

def get_face_box(net, frame, conf_threshold=0.7):
    frameCopy = frame.copy()
    h, w = frameCopy.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frameCopy,
        1.0,
        (300, 300),
        [104, 117, 123],
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            faceBoxes.append([x1, y1, x2, y2])

    return faceBoxes


# ===============================
# Age + Gender Prediction
# ===============================

def predict_age_gender(frame):
    faceBoxes = get_face_box(faceNet, frame)

    if not faceBoxes:
        return None, None

    x1, y1, x2, y2 = faceBoxes[0]
    face = frame[y1:y2, x1:x2]

    blob = cv2.dnn.blobFromImage(
        face,
        1.0,
        (227, 227),
        MODEL_MEAN_VALUES,
        swapRB=False
    )

    # Gender
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    # Age
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    return age, gender