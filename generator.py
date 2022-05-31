from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
import base64

DETECTOR_MODEL_INPUT_SIZE = (224, 224)
GENERATOR_MODEL_INPUT_SIZE = (128, 128)

def process_image_to_detector_input(face, shape):
    face = cv2.resize(face, shape)
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return face


def process_image_to_generator_input(face, shape):
    face = cv2.resize(face, shape)
    face = (np.asarray(face)) / 255
    face = np.expand_dims(face, axis=0)
    return face


def generate_img(face_to_generate):
    generator_model = load_model('./generator_model/generator-all-augs.h5')
    face_to_generate = process_image_to_generator_input(face_to_generate, GENERATOR_MODEL_INPUT_SIZE)
    prediction = generator_model.predict(face_to_generate)
    return prediction[0]


def find_faces(image):
    face_cascade = cv2.CascadeClassifier('./detector_model/haarcascade_frontalface_default.xml')
    return face_cascade.detectMultiScale(image, 1.1, 4)


def main(img):
    prototxtPath = './detector_model/deploy.prototxt.txt'
    weightsPath = './detector_model/res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

    detector_model = load_model('./detector_model/detectorModel')
    image = cv2.imread('./with-mask-default-mask-seed0006.png')
    img = img.decode('ascii')
    img = img.split(',')[1]
    nparr = np.fromstring(base64.b64decode(img), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    (h, w) = image.shape[:2]
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:

            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX- 10), max(0, startY-10))
            (endX, endY) = (min(w - 1, endX+10), min(h - 1, endY+10))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]

            face_to_detect = face.copy()
            face_to_generate = face.copy()

            face_to_detect = process_image_to_detector_input(face_to_detect, DETECTOR_MODEL_INPUT_SIZE)
            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = detector_model.predict(face_to_detect)[0]
            plt.imshow(face)
            plt.show()
            if mask > withoutMask:
                prediction = generate_img(face_to_generate)
                plt.imshow(prediction)
                plt.show()
                prediction_origin_size = cv2.resize(prediction, (face.shape[1], face.shape[0]))
                prediction_origin_size = prediction_origin_size * 255
                image[startY:endY, startX:endX] = prediction_origin_size


    plt.imshow(image)
    plt.show()
    buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # res_path = './result.png'  
    # cv2.imwrite(res_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return buffer