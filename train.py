"""Generate embedding and store in JSON"""
import os
import cv2
from utils.model_utils import faceDetector, faceAlignment, faceReconizer
from utils.data_utils import save_dummy_db

RAW_DIR_PATH = "data/raw"
FACE_DIR_PATH = "data/face"
DUMMY_DB = "data/dummy_db.json"

names = []
embeddings = []
for label in os.listdir(RAW_DIR_PATH):
    if label != ".DS_Store":
        print(f"Create embedding for {label}")
        for i, fn in enumerate(os.listdir(os.path.join(RAW_DIR_PATH, label))):
            img_path = os.path.join(RAW_DIR_PATH, label, fn)
            image = cv2.imread(img_path)
            boxes, labels, probs = faceDetector(image, 0.9)
            faces_, names_ = faceAlignment(boxes, image)
            # faces_, names_ = faceAlignment(boxes, image, FACE_DIR_PATH, label, i)
            embeddings_ = faceReconizer(faces_)

            names.append(names_[0])
            embeddings.append(embeddings_[0][0].tolist())

save_dummy_db(names, embeddings, DUMMY_DB)
