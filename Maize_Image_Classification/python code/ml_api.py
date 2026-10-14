import uvicorn
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
import os, joblib
from joblib import dump, load
import pickle
import pydantic
from io import BytesIO
from PIL import Image
import tensorflow as tf

# templates = Jinja2Templates(directory='templates')
app = FastAPI()

# pkl
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_dev_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, 'maize_img'))
model_path = os.path.join(ml_dev_dir, 'python code')
if os.path.exists(model_path):
    print('Changed directory to {}'.format(model_path))
else:
    print('Directory {} does not exist'.format(model_path))
    """with open('maize_disease_classification.pkl', 'rb') as file:
        model = pickle.load(file)"""

MODEL = tf.keras.models.load_model('../saved_model/1')
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']


# making predictions
@app.get("/ping")
async def ping():
    return {'message': 'Hello, world'}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
