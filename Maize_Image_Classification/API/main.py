# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:13:30 2023

@author: ADMIN
"""

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
import PIL
from PIL import Image
from keras.saving.saving_api import load_model
from pydantic import BaseModel
#from tensorflow.keras.models import load_model
import tensorflow, tensorflow_serving
from typing import List
import io
import numpy as np
import sys
import asyncio
import pickle

app = FastAPI()
pickel_in = open('.')

#load model
#filepath = '/saved_model'
#model = load_model(filepath, compile=True)

#get input shape for model layer
input_shape = model.layers[0].input_shape



app = FastAPI()

food_items = {
    'indian' : ['samosa', 'dosa'],
    'american' : ['hot dog', 'apple pie'],
    'italian' : ['ravioli', 'pizza']
}

@app.get('/get_items/{cuisine}')
async def get_items(cuisine):
    return food_items.get(cuisine)





