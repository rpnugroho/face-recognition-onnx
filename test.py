import cv2
import json
import numpy as np

from utils.data_utils import load_dummy_db
from utils.model_utils import imgPreprocesing, faceDetector, faceAlignment, faceReconizer

THRESHOLD = 0.33
DUMMY_DB = "data/dummy_db.json"


db_names, db_embeddings = load_dummy_db(DUMMY_DB)

video_capture = cv2.VideoCapture(0)
while True:
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    ret, frame = video_capture.read()

    # get boxes
    boxes, labels, probs = faceDetector(frame)
    # faces = []
    # boxes[boxes < 0] = 0
    # get faces
    faces, _ = faceAlignment(boxes, frame)

    if len(faces) > 0:
        predictions = []
        distances = []
        # get embeddings
        embeddings = faceReconizer(faces)
        # calculate distance for each embeddings
        for embedding in embeddings:
            distance = np.linalg.norm(db_embeddings - np.array(embedding), axis=1)
            # get sortest distance
            idx = np.argmin(distance)
            distances.append(distance[idx])
            if distance[idx] < THRESHOLD:
                predictions.append(db_names[idx])
            else:
                predictions.append("unknown")

        for i in range(boxes.shape[0]):
            # draw boxes
            x1, y1, x2, y2 = boxes[i, :]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 0), 3)
            # draw label
            text = f"{predictions[i]}: {format(distances[i], '.3f')}"
            cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), (0, 100, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.7, (255, 255, 255), 1)

    cv2.imshow("Video", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
