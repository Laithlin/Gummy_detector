import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
import os
import json
import time

def check_image(image):
    if image is None:
        sys.exit("Could not read the image")

# def detect_bears(image):
#     cascade_Path = "cascades/bears_cascade.xml"
#     pplCascade = cv.CascadeClassifier(cascade_Path)
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#     gray = cv.equalizeHist(gray)
#
#     pedestrians = pplCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.2,
#         minNeighbors=10,
#         minSize=(32, 96),
#         flags=cv.CASCADE_SCALE_IMAGE
#     )
#
#     # lower_red = np.array([130, 50, 50])
#     # upper_red = np.array([190, 255, 255])
#     # maska = cv.inRange(np.array([140, 0, 200]), lower_red, upper_red)
#     # print(maska[0])
#     # cv.imshow('dis', pedestrians[0])
#     # cv.waitKey(0)
#     # print(pedestrians[0])
#
#     return pedestrians

# def detect_circles(image):
#     cascade_Path = "cascades/circles_cascade.xml"
#     pplCascade = cv.CascadeClassifier(cascade_Path)
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#     gray = cv.equalizeHist(gray)
#
#     pedestrians = pplCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.2,
#         minNeighbors=10,
#         minSize=(32, 96),
#         flags=cv.CASCADE_SCALE_IMAGE
#     )
#
#     return pedestrians

def detect_worms(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2HLS)

    # Find Canny edges
    edged = cv.Canny(gray, 50, 200)
    cv.waitKey(0)
    maska = cv.inRange(image, np.array([100, 10, 10]), np.array([100, 255, 255]))
    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv.findContours(maska,
                                           cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)



    # scale = 0.2
    # width = int(image.shape[1] * scale)
    # height = int(image.shape[0] * scale)
    # dim = (width, height)
    #
    # # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    # resized_area = cv.resize(edged, dim, interpolation=cv.INTER_AREA)
    #
    # cv.imshow("Ppl found", resized_area)
    # cv.waitKey(0)

    # print("Number of Contours found = " + str(len(contours)))
    # for i in range(contours):
    #     print(len(i[0]))
    for i, c in enumerate(contours):
        area = cv.contourArea(c)
        if area > 100:
            print("numer: " + str(i))
            print(area)

    # pole = cv.inRange(contours[2], contours[1], contours[3])
    # print(pole)

    # Draw all contours
    # -1 signifies drawing all contours
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)

    scale = 0.2
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    cv.imshow("Ppl found", resized_area)
    cv.waitKey(0)

    return image

# def find_color(image, object):
#     x = object[0]
#     y = object[1]
#     w = object[2]
#     h = object[3]
#
#     crop_img = image[y:y + h, x:x + w]
#     mask_image = cv.cvtColor(crop_img, cv.COLOR_BGR2RGB)
#     # median = cv.medianBlur(crop_img, 5)
#     gray_image = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
#     ret, thresh = cv.threshold(gray_image, 127, 255, 0)
#     M = cv.moments(thresh)
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#
#
#     #TODO ogarniecie masek
#
#     # b = median[cX, cY, 0]
#     # g = median[cX, cY, 1]
#     # r = median[cX, cY, 2]
#     # rgb = np.array([int(r), int(g), int(b)])
#     # good = np.array([[255], [255], [255]])
#     # print(rgb)
#
#     b = crop_img[cX, cY, 0]
#     g = crop_img[cX, cY, 1]
#     r = crop_img[cX, cY, 2]
#     rgb = np.array([int(r), int(g), int(b)])
#     good = np.array([[255], [255], [255]])
#     print(rgb)
#
#     # TODO sprawdzanie jak sie wyswietla
#
#     # scale = 0.2
#     # width = int(image.shape[1] * scale)
#     # height = int(image.shape[0] * scale)
#     # dim = (width, height)
#     #
#     # # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
#     # resized_area = cv.resize(median, dim, interpolation=cv.INTER_AREA)
#     #
#     # cv.imshow("Ppl found", resized_area)
#     # cv.waitKey(0)
#
#     # cv.circle(crop_img, (cX, cY), 5, (255, 255, 255), -1)
#     #
#     # cv.imshow("Ppl found", crop_img)
#     # cv.waitKey(0)
#
#     lower_red = np.array([135, 28, 0])
#     upper_red = np.array([185, 70, 60])
#
#     maska_rl = cv.inRange(rgb, lower_red, upper_red)
#     # maska_rl = cv.inRange(np.array([140, 10, 0]), lower_red, upper_red)
#     # print(maska_rl)
#     if np.array_equal(maska_rl, good):
#         print("red")
#         mask = cv.inRange(mask_image, lower_red, upper_red)
#         bear_or_circle(crop_img, mask)
#
#     lower_red_dark = np.array([100, 0, 20])
#     upper_red_dark = np.array([146, 38, 55])
#     maska_rd = cv. inRange(rgb, lower_red_dark, upper_red_dark)
#     if np.array_equal(maska_rd, good):
#         print("dark red")
#         mask = cv.inRange(mask_image, lower_red_dark, upper_red_dark)
#         bear_or_circle(crop_img, mask)
#
#     lower_orange = np.array([148, 70, 0])
#     upper_orange = np.array([197, 121, 71])
#     maska_o = cv.inRange(rgb, lower_orange, upper_orange)
#     if np.array_equal(maska_o, good):
#         print("orange")
#         mask = cv.inRange(mask_image, lower_orange, upper_orange)
#         bear_or_circle(crop_img, mask)
#
#     lower_green = np.array([10, 68, 0])
#     upper_green = np.array([115, 145, 75])
#     maska_g = cv.inRange(rgb, lower_green, upper_green)
#     if np.array_equal(maska_g, good):
#         print("green")
#         mask = cv.inRange(mask_image, lower_green, upper_green)
#         bear_or_circle(crop_img, mask)
#
#     lower_yellow = np.array([150, 122, 0])
#     upper_yellow = np.array([190, 165, 71])
#     maska_y = cv.inRange(rgb, lower_yellow, upper_yellow)
#     if np.array_equal(maska_y, good):
#         print("yellow")
#         mask = cv.inRange(mask_image, lower_yellow, upper_yellow)
#         bear_or_circle(crop_img, mask)
#
#     lower_white = np.array([155, 115, 98])
#     upper_white = np.array([185, 175, 165])
#     maska_w = cv.inRange(rgb, lower_white, upper_white)
#     if np.array_equal(maska_w, good):
#         print("white")
#         mask = cv.inRange(mask_image, lower_white, upper_white)
#         bear_or_circle(crop_img, mask)
#
#     c = "none"
#     return c


# def write_to_file(file, img_name, bears, circles, worms):
#     # colors
#     # bears: light red(133, 0, 0), dark red(101,0,4),
#     # orange(169, 100, 0), green(50, 90, 13), yellow(158, 143, 0), white(159, 156, 99)
#     # circles: light red(147,9,0), dark red(112, 0, 33),
#     # orange(163, 107, 0), green(57, 107, 39), yellow(150, 158, 25), white(163, 167, 121)
#     # worms: green(10, 116, 0)-white(161, 175, 162), orange(156, 99, 11)-dark red(37, 13, 16),
#     # yellow(153, 156, 0)-red(187, 0, 0)
#     #file = open(str(path) + 'results.txt', 'w')
#
#     # output = {
#     #     img_name:
#     #         [
#     #             bears[0], bears[1], bears[2], bears[3], bears[4], bears[5],
#     #             circles[0], circles[1], circles[2], circles[3], circles[4], circles[5],
#     #             worms[0], worms[1], worms[2]
#     #         ]
#     # }
#
#     output = {
#         img_name:
#             [
#                 'dziala' + str(bears)
#             ]
#     }
#
#     out = json.dumps(output)
#     file.write(out)

# def bear_or_circle(ob, mask):
#     output_img = ob.copy()
#     output_img[np.where(mask == 0)] = 0
#     output_img[np.where(mask == 255)] = 255
#     # image = cv.cvtColor(output_img, cv.COLOR_BGR2RGB)
#     # size1 = output_img[np.where(mask == 255, [1], [0])]
#     size = output_img[np.where(mask == 255)]
#     # print(size)
#     # print(len(size1))
#
#     # for i in range(output_img.shape[0]):
#     #     for j in range(output_img.shape[1]):
#     #
#     kernel = np.ones((5, 5), np.uint8)
#     laplacian = cv.Laplacian(output_img, cv.CV_64F)
#     sobelx = cv.Sobel(ob, cv.CV_64F, 1, 0, ksize=5)
#     sobely = cv.Sobel(output_img, cv.CV_64F, 0, 1, ksize=5)
#
#
#     dilation = cv.dilate(laplacian, kernel, iterations=1)
#     closing = cv.morphologyEx(laplacian, cv.MORPH_CLOSE, kernel)
#
#     min = np.min(laplacian)
#     laplacian = laplacian - min  # to have only positive values
#     max = np.max(laplacian)
#     div = max / float(255)  # calculate the normalize divisor
#     lap = np.uint8(np.round(laplacian / div))
#
#     # x, y, w, h = cv.boundingRect(closing)
#     # cv.rectangle(closing, (x, y), (x + w, y + h), (255, 0, 0), 5)
#     size = closing[np.where(mask != 0)]
#     # print(len(size[0]))
#     cnts = cv.findContours(lap.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#
#     # print(np.where(output_img == 255, [1], [0]))
#     check = cv.bitwise_and(output_img, output_img, mask)
#     cv.imshow("Ppl found", closing)
#     cv.waitKey(0)

def bear_or_circle(box):
    print("ok")
    bears = 0
    circles = 0
    worm = 0

    for b in box:
        x, y = b[0]
        w, h = b[1]
        area = w*h
        print("Pole: " + str(area))
        if h > w:
            print("stosunek: " + str(h / w))
            compare = h/w
        else:
            print("stosunek: " + str(w / h))
            compare = w/h
        # area = w*h

        if area > 35000:
            worm += 1
            continue

        if compare > 2.0:
            bears +=1
        elif compare < 2.0:
            circles += 1


    colors = [bears, circles, worm]
    return colors

def check_if_repetitive(box, rec):
    same = False
    bigger = False
    which = box
    for r in rec:
        if (r[0][0] + 50 > box[0][0] > r[0][0] - 50) and (r[0][1] + 50 > box[0][1] > r[0][1] - 50):
            which = r
            print("poprzednie: " + str(r[0][0]))
            print("nowe" + str(rec[0][0]))
            same = True

        if (r[1][0] > box[1][0]) and (r[1][1] > box[1][1]):
            bigger = True
            which = box
            break

    check = [same, bigger, which]
    return check

def find_color(image):

    worm_g = 0
    worm_o = 0
    worm_y = 0
    blur = cv.blur(image, (30, 30))
    image = cv.blur(blur, (30, 30))
    image = cv.blur(image, (30, 30))
    image = cv.blur(image, (30, 30))
    image = cv.blur(image, (30, 30))
    #TODO ogarniecie masek

    # good = np.array([[255], [255], [255]])
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # TODO sprawdzanie jak sie wyswietla

    # scale = 0.2
    # width = int(image.shape[1] * scale)
    # height = int(image.shape[0] * scale)
    # dim = (width, height)
    #
    # # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    # resized_area = cv.resize(median, dim, interpolation=cv.INTER_AREA)
    #
    # cv.imshow("Ppl found", resized_area)
    # cv.waitKey(0)

    # cv.circle(crop_img, (cX, cY), 5, (255, 255, 255), -1)
    #
    # cv.imshow("Ppl found", crop_img)
    # cv.waitKey(0)
    #RGB
    # lower_red = np.array([135, 28, 0])
    # upper_red = np.array([185, 70, 60])

    # white_l = np.array([175, 199, 199])
    # white_u = np.array([255, 255, 255])
    #
    # white_low = np.array([145, 130, 110])
    # white_up = np.array([151, 145, 130])
    #
    # lower_red = white_low + white_l
    # upper_red = white_up + white_u

    # check = cv.bitwise_and(image, image, maska_rl)
    # laplacian = cv.Laplacian(check, cv.CV_8UC1)
    #
    # minVal = np.amin(laplacian)
    # maxVal = np.amax(laplacian)
    # lap = cv.convertScaleAbs(laplacian, alpha=255.0 / (maxVal - minVal), beta=-minVal * 255.0 / (maxVal - minVal))
    # print(cnts)
    # print(len(cnts))
    # x, y, w, h = cv.boundingRect(cnts[2])
    # cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    #TODO light red
    image_r = image.copy()
    # BGR
    lower_red_1 = np.array([20, 25, 140])
    upper_red_1 = np.array([65, 70, 165])
    maska_rl_1 = cv.inRange(image_r, lower_red_1, upper_red_1)
    lower_red_2 = np.array([15, 20, 112])
    upper_red_2 = np.array([47, 55, 135])
    maska_rl_2 = cv.inRange(image_r, lower_red_2, upper_red_2)
    lower_red_3 = np.array([22, 30, 155])
    upper_red_3 = np.array([50, 50, 182])
    maska_rl_3 = cv.inRange(image_r, lower_red_3, upper_red_3)

    maska_rl = maska_rl_1 + maska_rl_3 + maska_rl_2
    # image_r[np.where(maska_rl == 0)] = 255
    # image_r[np.where(maska_rl == 255)] = 0
    rec_r = []
    cnts_rl, hier_rl = cv.findContours(maska_rl, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours_poly_rl = [None] * len(cnts_rl)
    # boundRect_rl = [None] * len(cnts_rl)
    # for i, c in enumerate(cnts_rl):
    #     contours_poly_rl[i] = cv.approxPolyDP(c, 3, True)
    #     boundRect_rl[i] = cv.boundingRect(contours_poly_rl[i])
    #
    # for i in range(len(cnts_rl)):
    #     # cv2.drawContours(image, contours_poly, i, (0, 255, 0), thickness=2)
    #     pt1 = (int(boundRect_rl[i][0]), int(boundRect_rl[i][1]))
    #     pt2 = (int(boundRect_rl[i][0] + boundRect_rl[i][2]), int(boundRect_rl[i][1] + boundRect_rl[i][3]))
    #     if np.sqrt((pt2[1] - pt1[1]) * (pt2[0] - pt1[0])) < 100:
    #         continue
    #     # cv.rectangle(image, pt1, pt2, (0, 255, 0), 5)
    #     # print((pt2[1] - pt1[1]) * (pt2[0] - pt1[0]))
    #     rec_r.append([pt1, pt2])

    for c in cnts_rl:
        if len(c) < 200:
            continue
        rec = cv.minAreaRect(c)
        check = check_if_repetitive(rec, rec_r)

        if (check[0] is True) and (check[1] is True):
            rec_r.remove(check[2])
        elif (check[0] is True) and (check[1] is False):
            continue
        elif (check[0] is False) and (check[1] is False):
            rec_r.append(rec)
        rec_r.append(rec)

    reds = bear_or_circle(rec_r)
    print("RED: ")
    print(reds)

    # cv.drawContours(image, cnts, -1, (0, 255, 0), 3)
    # print(len(rec_r))
    # p1 = rec[11][0]
    # p2 = rec[11][-1]
    # cv.rectangle(image, p1, p2, (0, 255, 0), 5)
    # print((p2[0] - p1[0]) * (p2[-1] - p1[-1]))

    # scale = 0.2
    # width = int(image.shape[1] * scale)
    # height = int(image.shape[0] * scale)
    # dim = (width, height)
    #
    # # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    # resized_area = cv.resize(image_r, dim, interpolation=cv.INTER_AREA)
    #
    # cv.imshow("Ppl found", resized_area)
    # cv.waitKey(0)

    #TODO dark red

    image_dr = image.copy()

    lower_red_dark = np.array([25, 10, 80])
    upper_red_dark = np.array([55, 38, 146])

    # scale = 0.2
    # width = int(image.shape[1] * scale)
    # height = int(image.shape[0] * scale)
    # dim = (width, height)
    #
    # # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    #
    # cv.imshow("Ppl found", resized_area)
    # cv.waitKey(0)

    maska_rd = cv.inRange(image_dr, lower_red_dark, upper_red_dark)
    # image_dr[np.where(maska_rd == 0)] = 0
    # image_dr[np.where(maska_rd == 255)] = 255
    rec_dr = []
    cnts, hier = cv.findContours(maska_rd, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours_poly = [None] * len(cnts)
    # boundRect = [None] * len(cnts)
    # for i, c in enumerate(cnts):
    #     contours_poly[i] = cv.approxPolyDP(c, 3, True)
    #     boundRect[i] = cv.boundingRect(contours_poly[i])
    # for i in range(len(cnts)):
    #     # cv2.drawContours(image, contours_poly, i, (0, 255, 0), thickness=2)
    #     pt1 = (int(boundRect[i][0]), int(boundRect[i][1]))
    #     pt2 = (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3]))
    #     if np.sqrt((pt2[1] - pt1[1]) * (pt2[0] - pt1[0])) < 100:
    #         continue
    #     cv.rectangle(image_dr, pt1, pt2, (0, 255, 0), 5)
    #     # print((pt2[1] - pt1[1]) * (pt2[0] - pt1[0]))
    #     rec_dr.append([pt1, pt2])

    for c in cnts:
        if len(c) < 100:
            continue
        rec = cv.minAreaRect(c)
        check = check_if_repetitive(rec, rec_dr)

        if (check[0] is True) and (check[1] is True):
            rec_dr.remove(check[2])
        elif (check[0] is True) and (check[1] is False):
            continue
        elif (check[0] is False) and (check[1] is False):
            rec_dr.append(rec)
            # box = cv.boxPoints(rec)  # cv2.boxPoints(rect) for OpenCV 3.x
            # box = np.int0(box)

    # w, h = rec_dr[5][1]
    # print("wysokosc: " + str(h) + "szerokosc: " + str(w))
    # if h > w:
    #     print("stosunek: " + str(h / w))
    # else:
    #     print("stosunek: " + str(w / h))
    # box = cv.boxPoints(rec_dr[5])  # cv2.boxPoints(rect) for OpenCV 3.x
    # box = np.int0(box)
    # cv.drawContours(image_dr, [box], 0, (0, 255, 0), 5)

    dark_reds = bear_or_circle(rec_dr)

    print("DARK RED: ")
    print(dark_reds)

    # scale = 0.2
    # width = int(image.shape[1] * scale)
    # height = int(image.shape[0] * scale)
    # dim = (width, height)
    #
    # # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    # resized_area = cv.resize(image_dr, dim, interpolation=cv.INTER_AREA)
    #
    # cv.imshow("Ppl found", resized_area)
    # cv.waitKey(0)

    # TODO orange
    image_o = image.copy()
    lower_orange_1 = np.array([15, 60, 116])
    upper_orange_1 = np.array([36, 77, 150])
    maska_1 = cv.inRange(image_o, lower_orange_1, upper_orange_1)
    lower_orange_2 = np.array([18, 80, 150])
    upper_orange_2 = np.array([58, 117, 187])
    maska_2 = cv.inRange(image_o, lower_orange_2, upper_orange_2)
    lower_orange_3 = np.array([15, 59, 125])
    upper_orange_3 = np.array([35, 76, 150])
    maska_3 = cv.inRange(image_o, lower_orange_3, upper_orange_3)
    lower_orange_4 = np.array([26, 70, 141])
    upper_orange_4 = np.array([44, 98, 149])
    maska_4 = cv.inRange(image_o, lower_orange_4, upper_orange_4)
    lower_orange_5 = np.array([28, 100, 184])
    upper_orange_5 = np.array([82, 152, 215])
    maska_5 = cv.inRange(image_o, lower_orange_5, upper_orange_5)

    maska_o = maska_1 + maska_2 + maska_3 + maska_4 + maska_5
    # image_o[np.where(maska_o == 0)] = 255
    # image_o[np.where(maska_o == 255)] = 0
    rec_o = []
    cnts, hier = cv.findContours(maska_o, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours_poly = [None] * len(cnts)
    # boundRect = [None] * len(cnts)
    # for i, c in enumerate(cnts):
    #     contours_poly[i] = cv.approxPolyDP(c, 3, True)
    #     boundRect[i] = cv.boundingRect(contours_poly[i])
    # for i in range(len(cnts)):
    #     # cv2.drawContours(image, contours_poly, i, (0, 255, 0), thickness=2)
    #     pt1 = (int(boundRect[i][0]), int(boundRect[i][1]))
    #     pt2 = (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3]))
    #     if np.sqrt((pt2[1] - pt1[1]) * (pt2[0] - pt1[0])) < 100:
    #         continue
    #     # cv.rectangle(image, pt1, pt2, (0, 255, 0), 5)
    #     # print((pt2[1] - pt1[1]) * (pt2[0] - pt1[0]))
    #     rec_o.append([pt1, pt2])

    for c in cnts:
        if len(c) < 100:
            continue
        rec = cv.minAreaRect(c)
        check = check_if_repetitive(rec, rec_o)

        if (check[0] is True) and (check[1] is True):
            rec_o.remove(check[2])
        elif (check[0] is True) and (check[1] is False):
            continue
        elif (check[0] is False) and (check[1] is False):
            rec_o.append(rec)

    oranges = bear_or_circle(rec_o)

    # scale = 0.2
    # width = int(image.shape[1] * scale)
    # height = int(image.shape[0] * scale)
    # dim = (width, height)
    #
    # # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    # resized_area = cv.resize(image_o, dim, interpolation=cv.INTER_AREA)
    #
    # cv.imshow("Ppl found", resized_area)
    # cv.waitKey(0)

    print("ORANGES: ")
    print(oranges)

    #TODO green

    lower_green = np.array([0, 68, 10])
    upper_green = np.array([50, 145, 115])

    image_g = image.copy()
    maska_g = cv.inRange(image_g, lower_green, upper_green)
    # image_g[np.where(maska_g == 0)] = 255
    # image_g[np.where(maska_g == 255)] = 0
    rec_g = []
    cnts, hier = cv.findContours(maska_g, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours_poly = [None] * len(cnts)
    # boundRect = [None] * len(cnts)
    # for i, c in enumerate(cnts):
    #     contours_poly[i] = cv.approxPolyDP(c, 3, True)
    #     boundRect[i] = cv.boundingRect(contours_poly[i])
    # for i in range(len(cnts)):
    #     # cv2.drawContours(image, contours_poly, i, (0, 255, 0), thickness=2)
    #     pt1 = (int(boundRect[i][0]), int(boundRect[i][1]))
    #     pt2 = (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3]))
    #     if np.sqrt((pt2[1] - pt1[1]) * (pt2[0] - pt1[0])) < 100:
    #         continue
    #     # cv.rectangle(image, pt1, pt2, (0, 255, 0), 5)
    #     # print((pt2[1] - pt1[1]) * (pt2[0] - pt1[0]))
    #     rec_g.append([pt1, pt2])

    for c in cnts:
        if len(c) < 100:
            continue
        rec = cv.minAreaRect(c)
        check = check_if_repetitive(rec, rec_g)

        if (check[0] is True) and (check[1] is True):
            rec_g.remove(check[2])
        elif (check[0] is True) and (check[1] is False):
            continue
        elif (check[0] is False) and (check[1] is False):
            rec_g.append(rec)

    greens = bear_or_circle(rec_g)

    print("GREEN: ")
    print(greens)

    #TODO yellow

    lower_yellow = np.array([0, 130, 150])
    upper_yellow = np.array([71, 165, 190])

    image_y = image.copy()
    maska_y = cv.inRange(image_y, lower_yellow, upper_yellow)
    # image_y[np.where(maska_y == 0)] = 255
    # image_y[np.where(maska_y == 255)] = 0
    rec_y = []
    cnts, hier = cv.findContours(maska_y, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours_poly = [None] * len(cnts)
    # boundRect = [None] * len(cnts)
    # for i, c in enumerate(cnts):
    #     contours_poly[i] = cv.approxPolyDP(c, 3, True)
    #     boundRect[i] = cv.boundingRect(contours_poly[i])
    # for i in range(len(cnts)):
    #     # cv2.drawContours(image, contours_poly, i, (0, 255, 0), thickness=2)
    #     pt1 = (int(boundRect[i][0]), int(boundRect[i][1]))
    #     pt2 = (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3]))
    #     if np.sqrt((pt2[1] - pt1[1]) * (pt2[0] - pt1[0])) < 100:
    #         continue
    #     # cv.rectangle(image, pt1, pt2, (0, 255, 0), 5)
    #     # print((pt2[1] - pt1[1]) * (pt2[0] - pt1[0]))
    #     rec_y.append([pt1, pt2])

    for c in cnts:
        if len(c) < 100:
            continue
        rec = cv.minAreaRect(c)
        check = check_if_repetitive(rec, rec_y)

        if (check[0] is True) and (check[1] is True):
            rec_y.remove(check[2])
        elif (check[0] is True) and (check[1] is False):
            continue
        elif (check[0] is False) and (check[1] is False):
            rec_y.append(rec)

    yellows = bear_or_circle(rec_y)

    # scale = 0.2
    # width = int(image.shape[1] * scale)
    # height = int(image.shape[0] * scale)
    # dim = (width, height)
    #
    # # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    # resized_area = cv.resize(image_y, dim, interpolation=cv.INTER_AREA)
    #
    # cv.imshow("Ppl found", resized_area)
    # cv.waitKey(0)

    print("YELLOW: ")
    print(yellows)

    #TODO white
    image_w = image.copy()
    lower_white_1 = np.array([80, 115, 125])
    upper_white_1 = np.array([145, 165, 170])
    maska_w_1 = cv.inRange(image_w, lower_white_1, upper_white_1)

    lower_white_2 = np.array([123, 140, 155])
    upper_white_2 = np.array([126, 155, 164])
    maska_w_2 = cv.inRange(image_w, lower_white_2, upper_white_2)

    maska_w = maska_w_1 - maska_w_2
    # image_w[np.where(maska_w == 0)] = 0
    # image_w[np.where(maska_w == 255)] = 255
    rec_w = []
    cnts, hier = cv.findContours(maska_w, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours_poly = [None] * len(cnts)
    # boundRect = [None] * len(cnts)
    # for i, c in enumerate(cnts):
    #     contours_poly[i] = cv.approxPolyDP(c, 3, True)
    #     boundRect[i] = cv.boundingRect(contours_poly[i])
    #
    # for i in range(len(cnts)):
    #     # cv2.drawContours(image, contours_poly, i, (0, 255, 0), thickness=2)
    #     pt1 = (int(boundRect[i][0]), int(boundRect[i][1]))
    #     pt2 = (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3]))
    #     if np.sqrt((pt2[1] - pt1[1]) * (pt2[0] - pt1[0])) < 100:
    #         continue
    #     # cv.rectangle(image, pt1, pt2, (0, 255, 0), 5)
    #     # print((pt2[1] - pt1[1]) * (pt2[0] - pt1[0]))
    #     rec_w.append([pt1, pt2])

    for c in cnts:
        if len(c) < 100:
            continue
        rec = cv.minAreaRect(c)
        check = check_if_repetitive(rec, rec_w)

        if (check[0] is True) and (check[1] is True):
            rec_w.remove(check[2])
        elif (check[0] is True) and (check[1] is False):
            continue
        elif (check[0] is False) and (check[1] is False):
            rec_w.append(rec)

    whites = bear_or_circle(rec_w)

    scale = 0.2
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    # resized_area = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    resized_area = cv.resize(image_w, dim, interpolation=cv.INTER_AREA)

    cv.imshow("Ppl found", resized_area)
    cv.waitKey(0)

    print("WHITE: ")
    print(whites)

    c = [reds[0], dark_reds[0], oranges[0], greens[0], yellows[0], whites[0],
         reds[1], dark_reds[1], oranges[1], greens[1], yellows[1], whites[1],
         greens[2], oranges[2], yellows[2]]

    # c = []

    return c

def main():
    # IMG PATHS
    start = time.time()
    # imagePath1 = "/home/justyna/Studia/magisterka_sem1/SW/train/train/img_000.jpg"
    imagePath = str(sys.argv[1])
    resPath = str(sys.argv[2])
    # file = open(resPath, 'w')
    zapis = {}
    # m_r_l = 0
    # m_r_d = 0
    # m_g = 0
    # m_w = 0
    # m_o = 0
    # m_y = 0
    # c_r_l = 0
    # c_r_d = 0
    # c_g = 0
    # c_w = 0
    # c_o = 0
    # c_y = 0
    # w_g_w = 0
    # w_o_dr = 0
    # w_y_r = 0

    # image = cv.imread(imagePath1)

    # worm = detect_worms(image)
    # bears = detect_bears(image)

    # print
    # "Found {0} ppl!".format(len(bears))

    for name in os.listdir(imagePath):

        path = str(imagePath) + str(name)
        image = cv.imread(path)
        print(name)
        color = find_color(image)
        # b_c_det = detect_bears(image)
        # for b_c in b_c_det:
        #
        #     color = find_color(image, b_c)
        #     color = "none"
        #
        #     if color == "green":
        #         m_g +=1
        #     elif color == "red_light":
        #         m_r_l += 1
        #     elif color == "red_dark":
        #         m_r_d += 1
        #     elif color == "yellow":
        #         m_y += 1
        #     elif color == "orange":
        #         m_o += 1
        #     elif color == "white":
        #         m_w += 1
        file = open(resPath, 'w')
        zapis[name] = color
        z = json.dumps(zapis)
        file.write(z)
        file.close()


    # z = json.dumps(zapis)
    # file.write(z)

    # Draw a rectangle around the detected objects
    # for (x, y, w, h) in bears:
    #     crop_img = image[y:y+h, x:x+w]
    #     contours, hierarchy = cv.findContours(crop_img,
    #                                           cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #     cv.drawContours(crop_img, contours, -1, (0, 255, 0), 3)
    #     cv.imshow("Ppl found", crop_img)
    #     cv.waitKey(0)

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
    # file.close()
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()