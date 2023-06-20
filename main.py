from fastapi import FastAPI, File
import numpy as np
import uvicorn
import os
import cv2
from PIL import Image
import io
from io import BytesIO
import base64
from starlette.responses import StreamingResponse
import threading
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

app = FastAPI()

xml_path = ('./weight/haarcascade_frontalface_alt.xml')

#############################################################################

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs2", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    For local js, css swagger in AUO
    :return:
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


##############################################################################

@app.get("/")
def HelloWorld():
    return {"Hello": "World"}


@app.post("/FaceDetectionAsJSON/")
def FaceDetectionAsJSON(file: bytes = File(...)):
    # get image
    cv2_img = bytes_to_cv2image(file)

    # inference
    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier(xml_path)
    faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    print("faces_rect", faces_rect)

    # draw
    result = []
    if len(faces_rect) > 0:
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            result.append({
                "label": "face",
                "bbox": [int(x), int(y), int(w), int(h)]
            })

    base64_img = cv2image_to_base64(cv2_img)
    output_dict = {"data": base64_img, "detections": result}
    return output_dict


@app.post("/FaceDetectionAsJPG/")
def FaceDetectionAsJPG(file: bytes = File(...)):
    # get image
    cv2_img = bytes_to_cv2image(file)

    # inference
    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier(xml_path)
    faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    # draw
    if len(faces_rect) > 0:
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    res, im_jpg = cv2.imencode(".jpg", cv2_img)
    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpeg")


mobileNet = None
class_names = None


def init_mobile_net():
    global mobileNet, class_names
    weight_path = "./weight/MobileNetSSD_deploy.caffemodel"
    config_path = "./weight/MobileNetSSD_deploy.prototxt"
    mobileNet = cv2.dnn.readNetFromCaffe(config_path, weight_path)
    mobileNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    mobileNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
    pass


init_mobile_net()
lock_mobile_od = threading.Lock()


@app.post("/MobileNetDetectionAsJPG/")
def MobileNetDetectionAsJPG(file: bytes = File(...)):
    global mobileNet, class_names
    image_bgr = bytes_to_cv2image(file)
    H, W, channel = image_bgr.shape
    blob = cv2.dnn.blobFromImage(image_bgr, 0.007843, (W, H), 127.5)
    mobileNet.setInput(blob)
    lock_mobile_od.acquire()
    detections = mobileNet.forward()
    lock_mobile_od.release()
    thresh = 0.6
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > thresh:
            pred_label = class_names[int(detections[0, 0, i, 1])]
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            image_bgr = cv2.rectangle(image_bgr, (startX, startY), (endX, endY), (0, 255, 0), 2)
            image_bgr = cv2.putText(image_bgr,
                                    "{}{:.02f}".format(pred_label, confidence),
                                    (startX, startY - 20),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1,
                                    (0, 255, 0), 2)

    res, im_jpg = cv2.imencode(".jpg", image_bgr)
    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpeg")


@app.post("/MobileNetDetectionAsJSON/")
def MobileNetDetectionAsJSON(file: bytes = File(...)):
    # get image
    global mobileNet, class_names
    image_bgr = bytes_to_cv2image(file)
    H, W, channel = image_bgr.shape
    blob = cv2.dnn.blobFromImage(image_bgr, 0.007843, (W, H), 127.5)
    mobileNet.setInput(blob)
    lock_mobile_od.acquire()
    detections = mobileNet.forward()
    lock_mobile_od.release()
    new_detections = []
    thresh = 0.6
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > thresh:
            pred_label = class_names[int(detections[0, 0, i, 1])]
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            new_detections.append(
                {
                    "label": pred_label,
                    "conf": "{:.02f}".format(confidence),
                    "bbox": [int(startX), int(startY), int(endX - startX), int(endY - startY)]
                }
            )
    return {"detections": new_detections}


def cv2image_to_base64(cv2img):
    retval, buffer_img = cv2.imencode('.jpg', cv2img)
    base64_str = base64.b64encode(buffer_img)
    str_a = base64_str.decode('utf-8')
    return str_a


def bytes_to_cv2image(imgdata):
    cv2img = cv2.cvtColor(np.array(Image.open(BytesIO(imgdata))), cv2.COLOR_RGB2BGR)
    return cv2img


############################################################################################
# 除了 file 的輸入以外，另外輸入一個 Json 相關的設定
from fastapi import Form, File, UploadFile
import json

from pydantic import BaseModel
from typing import Optional


class StructureBase(BaseModel):
    text: str
    x: Optional[int] = 10
    y: Optional[int] = 120
    f_scaled: Optional[float] = 1

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
    # ----------------------------------------------------------


@app.post("/draw_text_as_jpg/")
def draw_text_as_jpg(data: StructureBase = Form(...), file: UploadFile = File(...)):
    """
    draw text by json parameters of data structure.
    output file
    """
    content = file.file.read()
    cvimg = bytes_to_cv2image(content)
    cv2.putText(cvimg, data.text, (data.x, data.y), cv2.FONT_HERSHEY_DUPLEX,
                data.f_scaled, (0, 255, 255), 1, cv2.LINE_AA)

    res, im_jpg = cv2.imencode(".jpg", cvimg)

    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpg")


@app.post("/draw_text_as_json/")
def draw_text_as_json(data: StructureBase = Form(...), file: UploadFile = File(...)):
    """
    draw text by json parameters of data structure.
    """
    content = file.file.read()
    cvimg = bytes_to_cv2image(content)
    cv2.putText(cvimg, data.text, (data.x, data.y), cv2.FONT_HERSHEY_DUPLEX,
                data.f_scaled, (0, 255, 255), 1, cv2.LINE_AA)

    str_base64 = cv2image_to_base64(cvimg)

    return {"data_input": data.dict(), "base64": str_base64}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)
