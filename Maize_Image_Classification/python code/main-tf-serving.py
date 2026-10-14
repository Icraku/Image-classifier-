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
import requests

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

endpoint = "http://localhost:8501/v1/maize_img/saved_model:predict"
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

    json_data = {
        'instances': img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    #prediction = response.json()['predictions'][0]
    pass



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# m_d_model = open('maize_disease_classification.pkl', 'rb')
# m_d_model_ls = joblib.load(m_d_model)

# ML Aspect
# @app.get("/")
# async def read_root(request: Request):
# return templates.TemplateResponse("base.html", {"request": request})


# @app.get("/predict/")
# async def make_prediction(request: Request, L1:float, W1:float, L2:float, W2:float):
"""testData = np.array([L1,W1,L2,W2]).reshape(-1,4)
    probalities = loaded_model.predict_proba(testData)[0]
    predicted = np.argmax(probalities)
    probabilty = probalities[predicted]
    predicted = labelsname[predicted]"""
# return templates.TemplateResponse("prediction.html",
# {"request": request,
# }
# )


# if __name__ == '__main__':
# uvicorn.run(app,host="127.0.0.1",port=8000)
