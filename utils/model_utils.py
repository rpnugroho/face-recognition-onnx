# scale current rectangle to box
# function to import model (detection, alignment, recognition)
import os
import cv2
import dlib
import numpy as np
import onnxruntime as ort
from imutils.face_utils.facealigner import FaceAligner
from utils.box_utils import predict, scale

# face detection model
face_detector_onnx = "models/version-RFB-320.onnx"
face_detector = ort.InferenceSession(face_detector_onnx)
# face alignment model
shape_predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
face_aligner = FaceAligner(
    shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3)
)
# face recognition model
face_recognizer_onnx = "models/mobilefacenet.onnx"
face_recognizer = ort.InferenceSession(face_recognizer_onnx)


def imgPreprocesing(orig_image):
    image_mean = np.array([127, 127, 127])
    image = (orig_image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image


# face detection method
def faceDetector(orig_image, threshold=0.7):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image = imgPreprocesing(image)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(
        orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold
    )
    return boxes, labels, probs


def faceAlignment(
    boxes, orig_image, face_dir_path=None, person_folder=None, image_count=0
):
    person_faces = []
    person_names = []

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = scale(boxes[i, :])
        gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        aligned_face = face_aligner.align(
            orig_image, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
        )
        aligned_face = cv2.resize(aligned_face, (112, 112))

        if face_dir_path is not None:
            directory = f"{face_dir_path}/{person_folder}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            face_data_path = f"{directory}/{image_count}_{i}.jpg"
            cv2.imwrite(face_data_path, aligned_face)
            person_names.append(person_folder)

        aligned_face = aligned_face - 127.5
        aligned_face = aligned_face * 0.0078125
        person_faces.append(aligned_face)

    return person_faces, person_names


def faceReconizer(person_faces):
    person_embeddings = []
    for face in person_faces:
        face = imgPreprocesing(face)
        input_name = face_recognizer.get_inputs()[0].name
        embedding = face_recognizer.run(None, {input_name: face})[0]
        person_embeddings.append(embedding)

    return person_embeddings
