from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
    generator_model = load_model('C:/dev/mask-off/server/generator_model/generator-all-augs.h5')
    face_to_generate = process_image_to_generator_input(face_to_generate, GENERATOR_MODEL_INPUT_SIZE)
    prediction = generator_model.predict(face_to_generate)
    return prediction[0]


def find_faces(image):
    face_cascade = cv2.CascadeClassifier('C:/dev/mask-off/server/detector_model/haarcascade_frontalface_default.xml')
    return face_cascade.detectMultiScale(image, 1.1, 4)


def main():
    detector_model = load_model('C:/dev/mask-off/server/detector_model/detectorModel')
    # image = cv2.imread('C:/mask/with_mask/with-mask-default-mask-seed0000.png')
    image = cv2.imread('C:/mask/maksssksksss111.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = find_faces(image)

    # loop over the detections
    for (x, y, w, h) in faces:
        # compute the (x, y)-coordinates of the bounding box for
        # the object
        (startX, startY, endX, endY) = (x, y, x + w, y + h)

        # extract the face ROI, convert it from BGR to RGB channel
        # ordering, resize it to 224x224, and preprocess it
        face = image[startY:endY, startX:endX]
        face_to_detect = face.copy()
        face_to_generate = face.copy()

        face_to_detect = process_image_to_detector_input(face_to_detect, DETECTOR_MODEL_INPUT_SIZE)

        # pass the face through the model to determine if the face
        # has a mask or not
        (mask, withoutMask) = detector_model.predict(face_to_detect)[0]

        if mask > withoutMask:
            prediction = generate_img(face_to_generate)
            plt.imshow(prediction)
            plt.show()
            prediction_origin_size = cv2.resize(prediction, (face.shape[0], face.shape[1]))
            prediction_origin_size = prediction_origin_size * 255
            image[startY:endY, startX:endX] = prediction_origin_size

    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main()
