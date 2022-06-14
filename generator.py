from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import base64

from tensorflow.keras.preprocessing.image import load_img
from numpy import expand_dims

DETECTOR_MODEL_INPUT_SIZE = (224, 224)
GENERATOR_MODEL_INPUT_SIZE = (128, 128)
BUFFER_PERCENTAGE = 0.05


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


# load an image
def load_image(filename, size=(256, 256)):
    # load image with the preferred size
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    # reshape to 1 sample
    pixels = expand_dims(pixels, 0)
    return pixels


def generate_img_p2p(face_to_generate):
    # save photo
    face_path = './face_to_generate.png'
    cv2.imwrite(face_path, cv2.cvtColor(face_to_generate, cv2.COLOR_RGB2BGR))
    # load source image
    src_image = load_image(face_path)
    print('Loaded', src_image.shape)
    # load model
    model = load_model('./generator_model/model_080001.h5')
    # generate image from source
    gen_image = model.predict(src_image)
    # scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0
    # plot the image
    return gen_image[0]


def find_faces(image):
    face_cascade = cv2.CascadeClassifier('./detector_model/haarcascade_frontalface_default.xml')
    return face_cascade.detectMultiScale(image, 1.1, 4)


def is_polygon_overlapped(first_polygon, second_start_coordinates):
    (startX, startY, endX, endY) = first_polygon
    (x, y) = second_start_coordinates
    return startX < x < endX and startY < y < endY


def proccing_photo(img, needPhotoProccing):
    prototxtPath = './detector_model/deploy.prototxt.txt'
    weightsPath = './detector_model/res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

    detector_model = load_model('./detector_model/detectorModel')
    # image = cv2.imread('./with-mask-default-mask-seed0006.png')
    if needPhotoProccing:
        nparr = np.fromstring(base64.b64decode(img), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        image = img

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = image.copy()

    (h, w) = image.shape[:2]
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    detections = detections[0, 0]
    detections = list(filter(lambda detection: detection[2] > 0.3, detections))
    detections.sort(key=lambda detection: detection[2], reverse=True)
    masks_polygons = []

    # loop over the detections
    for strong_detection in detections:
        print("Face Detected!")
        # compute the (x, y)-coordinates of the bounding box for
        # the object
        box = strong_detection[3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        any_parent_mask = any(is_polygon_overlapped(mask, (startX, startY)) for mask in masks_polygons)
        if not any_parent_mask:
            original_coordinates = (startX, startY, endX, endY)
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            # (startX, startY) = (max(0, startX-10), max(0, startY-10))
            # (endX, endY) = (min(w - 1, endX+10), min(h - 1, endY+10))
            startX = int(max(startX - w * BUFFER_PERCENTAGE, 0))
            startY = int(max(startY - h * BUFFER_PERCENTAGE, 0))
            endX = int(min(endX + w * BUFFER_PERCENTAGE, w))
            endY = int(min(endY + h * BUFFER_PERCENTAGE, h))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]

            if len(face) > 0:
                face_to_detect = face.copy()
                face_to_generate = face.copy()

                face_to_detect = process_image_to_detector_input(face_to_detect, DETECTOR_MODEL_INPUT_SIZE)
                # pass the face through the model to determine if the face
                # has a mask or not
                (mask, withoutMask) = detector_model.predict(face_to_detect)[0]
                if mask > withoutMask:
                    print("Mask Detected!")
                    prediction = generate_img_p2p(face_to_generate)
                    prediction_origin_size = cv2.resize(prediction, (face.shape[1], face.shape[0]))
                    prediction_origin_size = prediction_origin_size * 255
                    image_copy[startY:endY, startX:endX] = prediction_origin_size
                    masks_polygons.append(original_coordinates)

    image = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
    if not needPhotoProccing:
        return image

    buffer = cv2.imencode('.png', image)
    return buffer


def check_if_video(file):
    filetype = file.split(',')[0]
    print(filetype)
    if 'video' in filetype:
        return True
    else:
        return False


def main(file):
    file = file.decode('ascii')
    if check_if_video(file):
        file = file.split(',')[1]
        input_path = 'input.mp4'
        with open(input_path, "wb") as fh:
            fh.write(base64.b64decode(file))

        # nparr = np.fromstring(base64.b64decode(file), np.uint8)
        # buffer = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # print(nparr)
        # # buffer = cv2.cvtColor(buffer, cv2.COLOR_BGR2RGB)  
        # # cv2.imwrite(input_path, cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR))

        # Creating a VideoCapture object to read the video
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('output.mp4', -1, fps, (frame_width, frame_height))
        DROP = 3
        dropCount = 0
        stuckcount = 0
        # Loop until the end of the video
        while (cap.isOpened()):
            flag, frame = cap.read()
            if flag:
                if dropCount % DROP == 0:
                    # The frame is ready and already captured
                    frame = proccing_photo(frame, False)
                    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    for x in range(0, DROP):
                        writer.write(frame)
                    print(str(pos_frame) + " frames")
                dropCount += 1
            else:
                if stuckcount > 2:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                # The next frame is not ready, so we try to read it again
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)
                stuckcount += 1
            if cv2.waitKey(10) == 27:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break

        # release the video capture object
        cap.release()
        writer.release()
        # Closes all the windows currently opened.
        # cv2.destroyAllWindows()

        with open("output.mp4", "rb") as videoFile:
            return videoFile.read()

    else:
        file = file.split(',')[1]
        res_photo = proccing_photo(file, True)[1]
        return res_photo
