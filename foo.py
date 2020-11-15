import cv2
from skimage import io
import numpy as np


def url_to_image(url):
    return io.imread(url)


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
        mouseX, mouseY = x, y


img = cv2.cvtColor(url_to_image('https://raw.githubusercontent.com/rpartsey/ucu-cv-2020/master/images/building/IMG_1265.JPG'), cv2.COLOR_BGR2RGB)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)


while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(mouseX, mouseY)