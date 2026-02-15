#!/usr/bin/env python
# coding: utf-8

# In[2]:


import io
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


# In[1]:


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import librosa
import io

# =========================
# Configuration
# =========================
SR = 22050
MFCC_N = 40
MAX_LEN = 500

# =========================
# Load Model (lighter version)
# =========================
model = tf.keras.models.load_model("sound_model_no_optimizer.keras")  # هنا استبدلنا الموديل القديم

app = FastAPI(title="Pneumonia Detection API")

# =========================
# MFCC Extraction Function
# =========================
def extract_mfcc_from_bytes(file_bytes):
    y, _ = librosa.load(io.BytesIO(file_bytes), sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=MFCC_N)
    mfcc = mfcc.T

    if mfcc.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:MAX_LEN, :]

    return mfcc

# =========================
# Prediction Endpoint
# =========================
@app.post("/predict")
async def predict(AudioRecord: UploadFile = File(...)):
    try:
        # قراءة الصوت
        contents = await AudioRecord.read()

        # استخراج MFCC
        mfcc = extract_mfcc_from_bytes(contents)

        # batch dimension
        mfcc = np.expand_dims(mfcc, axis=0)

        # prediction
        prediction = model.predict(mfcc)[0][0]

        # تحويل ل 0 أو 1
        result = 1 if prediction >= 0.5 else 0

        return {
            "prediction": result,
            "probability": float(prediction)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# In[ ]:




