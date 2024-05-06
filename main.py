import numpy as np
import cv2
import io
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pathlib import Path

import end2end
app = FastAPI()

@app.get("/")
async def read_root():
    return {'helloworld'}

@app.post("/inference")
async def inf(image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))
    # image = Path((io.BytesIO(content)))
    image = np.array(image)
    time, res_lp, img_draw_detect, inp_recog, image_draw_recog = end2end.end2end(image)
    return {"img_show": img_draw_detect}