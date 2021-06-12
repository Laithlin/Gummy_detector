import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
import json

def check_image(image):
    if image is None:
        sys.exit("Could not read the image")

def detect_bears(image):
    cascade_Path = "cascades/bears_cascade.xml"
    pplCascade = cv.CascadeClassifier(cascade_Path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    gray = cv.equalizeHist(gray)

    pedestrians = pplCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(32, 96),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # cv.imshow('dis', pedestrians[0])
    # cv.waitKey(0)
    print(pedestrians[0])

    return pedestrians

def detect_circles(image):
    cascade_Path = "cascades/circles_cascade.xml"
    pplCascade = cv.CascadeClassifier(cascade_Path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    gray = cv.equalizeHist(gray)

    pedestrians = pplCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(32, 96),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    return pedestrians

def detect_worms(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2HLS)

    # Find Canny edges
    edged = cv.Canny(gray, 50, 200)
    cv.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv.findContours(edged,
                                           cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    scale = 0.2
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    resized_area = cv.resize(edged, dim, interpolation=cv.INTER_AREA)

    cv.imshow("Ppl found", resized_area)
    cv.waitKey(0)

    print("Number of Contours found = " + str(len(contours)))
    # for i in range(contours):
    #     print(len(i[0]))


    # pole = cv.inRange(contours[2], contours[1], contours[3])
    # print(pole)

    # Draw all contours
    # -1 signifies drawing all contours
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)

    return image

def find_color(image, objects):
    color = []
    for (x, y, w, h) in objects:
        b = image[x + w/2, y + h/2, 0]
        g = image[x + w/2, y + h/2, 1]
        r = image[x + w/2, y + h/2, 2]
        color.append([b, g, r])

    return color


def write_to_file(path, img_name, bears, circles, worms):
    # colors
    # bears: light red, dark red, orange, green, yellow, white
    # circles: light red, dark red, orange, green, yellow, white
    # worms: green-white, orange-dark red, yellow-red
    file = open('results.txt', 'w')

    output = {
        img_name:
            [
                bears[0], bears[1], bears[2], bears[3], bears[4], bears[5],
                circles[0], circles[1], circles[2], circles[3], circles[4], circles[5],
                worms[0], worms[1], worms[2]
            ]
    }

    file.write(output)
    file.close()

def main():
    # IMG PATHS
    imagePath = "img_000.jpg"

    image = cv.imread(imagePath)

    # worm = detect_worms(image)
    bears = detect_bears(image)

    # print
    # "Found {0} ppl!".format(len(bears))

    # Draw a rectangle around the detected objects
    for (x, y, w, h) in bears:
        crop_img = image[y:y+h, x:x+w]
        contours, hierarchy = cv.findContours(crop_img,
                                              cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(crop_img, contours, -1, (0, 255, 0), 3)
        cv.imshow("Ppl found", crop_img)
        cv.waitKey(0)

    # cv.imwrite("saida.jpg", image)

    # scale = 0.2
    # width = int(image.shape[1] * scale)
    # height = int(image.shape[0] * scale)
    # dim = (width, height)
    #
    # # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    # resized_area = cv.resize(worm, dim, interpolation=cv.INTER_AREA)

    # cv.imshow("Ppl found", resized_area)
    # cv.waitKey(0)


if __name__ == '__main__':
    main()