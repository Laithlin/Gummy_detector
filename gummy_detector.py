import numpy as np
import cv2 as cv
import sys
import os
import json
import time

def check_image(image):
    if image is None:
        sys.exit("Could not read the image")

def bear_or_circle(box):
    # print("ok")
    bears = 0
    circles = 0
    worm = 0

    for b in box:
        x, y = b[0]
        w, h = b[1]
        area = w*h
        # print("Pole: " + str(area))
        if h > w:
            # print("stosunek: " + str(h / w))
            compare = h/w
        else:
            # print("stosunek: " + str(w / h))
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
            # print("poprzednie: " + str(r[0][0]))
            # print("nowe" + str(rec[0][0]))
            same = True

        if (r[1][0] > box[1][0]) and (r[1][1] > box[1][1]):
            bigger = True
            which = box
            break

    check = [same, bigger, which]
    return check

def find_color(image):

    blur = cv.blur(image, (30, 30))
    image = cv.blur(blur, (30, 30))
    image = cv.blur(image, (30, 30))
    image = cv.blur(image, (30, 30))
    image = cv.blur(image, (30, 30))

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
    rec_r = []
    cnts_rl, hier_rl = cv.findContours(maska_rl, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

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

    reds = bear_or_circle(rec_r)
    # print("RED: ")
    # print(reds)

    #TODO dark red

    image_dr = image.copy()

    lower_red_dark = np.array([25, 10, 80])
    upper_red_dark = np.array([55, 38, 146])

    maska_rd = cv.inRange(image_dr, lower_red_dark, upper_red_dark)
    rec_dr = []
    cnts, hier = cv.findContours(maska_rd, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

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

    dark_reds = bear_or_circle(rec_dr)

    # print("DARK RED: ")
    # print(dark_reds)

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
    rec_o = []
    cnts, hier = cv.findContours(maska_o, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

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

    # print("ORANGES: ")
    # print(oranges)

    #TODO green

    lower_green = np.array([0, 68, 10])
    upper_green = np.array([50, 145, 115])

    image_g = image.copy()
    maska_g = cv.inRange(image_g, lower_green, upper_green)
    rec_g = []
    cnts, hier = cv.findContours(maska_g, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

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

    # print("GREEN: ")
    # print(greens)

    #TODO yellow

    lower_yellow = np.array([0, 130, 150])
    upper_yellow = np.array([71, 165, 190])

    image_y = image.copy()
    maska_y = cv.inRange(image_y, lower_yellow, upper_yellow)
    rec_y = []
    cnts, hier = cv.findContours(maska_y, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

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

    # print("YELLOW: ")
    # print(yellows)

    #TODO white
    image_w = image.copy()
    lower_white_1 = np.array([80, 115, 125])
    upper_white_1 = np.array([145, 165, 170])
    maska_w_1 = cv.inRange(image_w, lower_white_1, upper_white_1)

    lower_white_2 = np.array([123, 140, 155])
    upper_white_2 = np.array([126, 155, 164])
    maska_w_2 = cv.inRange(image_w, lower_white_2, upper_white_2)

    maska_w = maska_w_1 - maska_w_2
    rec_w = []
    cnts, hier = cv.findContours(maska_w, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

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

    # print("WHITE: ")
    # print(whites)

    c = [reds[0], dark_reds[0], oranges[0], greens[0], yellows[0], whites[0],
         reds[1], dark_reds[1], oranges[1], greens[1], yellows[1], whites[1],
         greens[2], oranges[2], yellows[2]]

    return c

def main():
    # IMG PATHS
    # start = time.time()
    imagePath = str(sys.argv[1])
    resPath = str(sys.argv[2])
    zapis = {}

    for name in os.listdir(imagePath):
        path = str(imagePath) + str(name)
        image = cv.imread(path)
        # print(name)
        color = find_color(image)
        file = open(resPath, 'w')
        zapis[name] = color
        z = json.dumps(zapis)
        file.write(z)
        file.close()

    # end = time.time()
    # print(end - start)

if __name__ == '__main__':
    main()