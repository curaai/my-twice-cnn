import cv2
from PIL import Image
import numpy as np
import os
import json
import requests
from scipy.misc import imread

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
client_id = "NgyeOZXlBMA1XytCZjZf"
client_secret = "cGSR9XEp5b"
url = "https://openapi.naver.com/v1/vision/face"
headers = {'X-Naver-Client-Id': client_id, 'X-Naver-Client-Secret': client_secret }

count = 0
IMAGE_SIZE = 96
IMAGE_LIMIT = 1300

def convert_video(input_path, output_path):
    global count

    vid = cv2.VideoCapture(input_path)
    success, image = vid.read()
    success = True
    while success and count < IMAGE_LIMIT:
        count += 1

        success, image = vid.read()
        if success:
            faces = face_detect(image)
            if faces == -1:
                continue
            image = np.array(Image.fromarray(image).crop(faces))
            cv2.imwrite(output_path + "frame%d.jpg" % count, image)


def face_detect_crop(input_path, output_path):
    image = cv2.imread(input_path)
    array = imread(input_path)
    faces = face_detect_by_naver(array)
    image = np.array(Image.fromarray(image).crop(faces))
    cv2.imwrite(output_path, image)

    image = Image.open(output_path)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    image.save(output_path)


def face_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return -1
    x, y, w, h = faces[0]
    return x, y, x + w, y + h


def face_detect_by_naver(image):
    cv2.imwrite('temp.jpg', image)
    byte_image = {'image': open('temp.jpg', 'rb')}
    response = requests.post(url, files=byte_image, headers=headers)
    rescode = response.status_code
    if rescode == 200:
        try:
            face = json.loads(response.text)['faces'][0]['roi']
        except IndexError:
            return -1

        x = face['x']
        y = face['y']
        h = face['height']
        w = face['width']

        return x, y, x + w, y + h


def image_resize(path):
    files = [path + '/' + directory + '/' + file for directory in os.listdir(path) for file in os.listdir(path + '/' + directory)]
    for file in files:
        image = Image.open(file)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        image.save(file)

if __name__ == '__main__':
    face_detect_crop("C:/Users/dsm2016/Desktop/go.jpg", 'C:/Users/dsm2016/Desktop/test1.jpg')
    face_detect_crop("C:/Users/dsm2016/Desktop/park.jpg", 'C:/Users/dsm2016/Desktop/test2.jpg')