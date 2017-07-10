import cv2
from PIL import Image
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def convert_video(input_path, output_path):
    vid = cv2.VideoCapture(input_path)
    success, image = vid.read()
    count = 0
    success = True
    while success:
        count += 1

        success, image = vid.read()
        if success:
            faces = face_detect(image)
            image = np.array(Image.fromarray(image).crop(faces))
            cv2.imwrite(output_path + "frame%d.jpg" % count, image)


def face_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    x, y, w, h = faces[0]
    return x, y, x + w, y + h

# test
# if __name__ == '__main__':
#     video_path = "C:/Users/dsm2016/Pictures/Camera Roll/WIN_20170710_18_13_07_Pro.mp4"
#     save_path = "C:/Users/dsm2016/Pictures/faces/"
#     convert_video(video_path, save_path)
