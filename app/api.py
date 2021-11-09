import cv2
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from utils.model_utils import faceDetector, faceAlignment, faceReconizer

FACE_DIR_PATH = "app/face"

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def predict_api(name: str, file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    image_file = await file.read()
    nparr = np.fromstring(image_file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # expecting only one person in image
    box, _, _ = faceDetector(image, 0.9)
    face, _ = faceAlignment(box, image)  # , FACE_DIR_PATH, "api", 0) #uncomment for debug
    embedding = faceReconizer(face)
    return {
        "name": name,
        "embedding": embedding[0][0].tolist(),
    }


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
