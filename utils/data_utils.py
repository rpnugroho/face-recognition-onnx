import numpy as np
import json


def save_dummy_db(names, embeddings, json_path):
    dummy_db = {
        "names": names,
        "embeddings": embeddings,
    }

    with open(json_path, "w") as f:
        json.dump(dummy_db, f)


def load_dummy_db(json_path):
    with open(json_path, "rb") as f:
        dummy_db = json.load(f)

    db_names = np.array(dummy_db["names"])
    db_embeddings = np.array(dummy_db["embeddings"])

    return db_names, db_embeddings
