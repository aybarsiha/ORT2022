#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:29:06 2022

@author: mr4t
"""
import numpy as np
import cv2
import math
from skimage.color import rgb2hsv, rgb2gray
from skimage.transform import resize
from skimage.io import imread
from skimage.transform import rotate
from imutils import rotate_bound
from tqdm import tqdm
import os
import time

import matplotlib.pyplot as plt

import tflite_runtime.interpreter as tflite
import tensorflow as tf

class ODTU():
    def __init(self):
        self.center = None
        self.angle = None
        self.image = None
        self.thresh = None
        self.lineCenter = None
        self.circle_params = None
        self.res = None
        self.number_pred = None
        self.shape_pred = None
        self.number_detector = None
        self.shape_detector = None
        self.last = None
    def shape_detection(self):
        self.shape_detector = tf.keras.models.load_model("shape_modelv33.h5")
    
    def number_detection(self):
        self.number_detector = tf.keras.models.load_model("ocr_mobileNet.h5")
        
        
    def prepareImage(self, image):
        self.image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
        hsv = (rgb2hsv(self.image)*255).astype(np.uint8)
        
        lowerB = np.array([0, 0, 0])
        upperB = np.array([255, 255, 45])
        
        self.thresh = cv2.inRange(hsv, lowerB, upperB)
        
    def detectShape(self):
        self.angle = None
        self.center = None
        self.circle_params = None
        self.res = None
        cnts, _ = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) != 0:
            cnt = cnts[np.argmax([cv2.contourArea(cnt) for cnt in cnts])]
            
            approx = cv2.approxPolyDP(cnt, 0.025 * cv2.arcLength(cnt, True), True)
            M = cv2.moments(cnt)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                
            def find_tip(points, convex_hull):
                length = len(points)
                indices = np.setdiff1d(range(length), convex_hull)
    
                for i in range(2):
                    j = indices[i] + 2
                    if j > length - 1:
                        j = length - j
                    if np.all(points[j] == points[indices[i - 1] - 2]):
                        return tuple(points[j])
                
            cnts, _ = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) != 0:
                last_cnts = cnts[np.argmax([cv2.contourArea(cnt) for cnt in cnts])]
            else:
                last_cnts = None
            if last_cnts is not None:
                peri = cv2.arcLength(last_cnts, True)
                approx = cv2.approxPolyDP(last_cnts, 0.025 * peri, True)
                hull = cv2.convexHull(approx, returnPoints=False)
                sides = len(hull)
    
    
                if 6 > sides > 3 and sides + 2 == len(approx):
                    # if list(self.last.values())[0][0] == "arrow":
                    #     self.number_pred = None
                    #     return 0
                    arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
                    if arrow_tip:
                        self.angle = (math.atan2(y-arrow_tip[1], x-arrow_tip[0])* 180/math.pi)
                        cv2.line(self.image, arrow_tip, [x,y], (0, 0, 255), 2)
                        # rotate(self.image, self.angle)
                        self.res = "arrow"
                        temp = np.zeros_like(self.thresh)

                        self.temp = cv2.drawContours(self.thresh, [last_cnts], 0, 0, -1)  
                        self.temp = cv2.dilate(self.temp, np.ones((11, 11)))
                        cnts, _ = cv2.findContours(self.temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        nearest = []
                        if len(cnts) != 0:
                            for cnt in cnts:
                                M = cv2.moments(cnt)
                                if M['m00'] != 0.0:
                                    x2 = int(M['m10']/M['m00'])
                                    y2 = int(M['m01']/M['m00'])
                                else:
                                    x2, y2 = 0, 0
                                nearest.append(((y2-y)**2 + (x2-x)**2)**1/2)
                            last_cnts = cnts[np.argmin(nearest)]
                            x, y, w, h = cv2.boundingRect(last_cnts)
                            # for last_cnts in cnts:
                            # cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0))
                            self.numberr = self.thresh[y:y+h, x:x+w].copy()
                            temp = np.zeros((max(self.numberr.shape), max(self.numberr.shape)))

                            temp[(temp.shape[0]//2)-(h//2):(temp.shape[0]//2)-(h//2)+h, (temp.shape[0]//2)-(w//2):(temp.shape[0]//2)-(w//2)+w] +=self.numberr
                            self.numberr = rotate(temp, self.angle-90)

                            self.numberr = self.numberr.astype(np.uint8)
                            _, self.numberr = cv2.threshold(self.numberr, 30, 255, cv2.THRESH_BINARY)
                            areas = {}
                            cnts, _ = cv2.findContours(self.numberr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in cnts:
                                areas[cv2.contourArea(cnt)] = cnt
                                
                                
                            coords = {}
                            # self.numberDetection()

                            for i in sorted(areas, reverse=True)[:3]:
                                x, y, w, h = cv2.boundingRect(areas[i])
                                coords[x] = self.numberr[y:y+h, x:x+w]
                            self.number_pred = ""
                            for coord in sorted(coords):
                                # img = np.stack((coords[coord], coords[coord], coords[coord]), axis=-1)
                                img = coords[coord]
                                prob = max(img.shape) / 20
                                img = cv2.resize(img, (int(img.shape[1] / prob), int(img.shape[0] / prob)))
                                temp = np.zeros((28, 28), np.uint8)
                                temp[(28-img.shape[0])//2:((28-img.shape[0])//2)+img.shape[0], (28-img.shape[1])//2:((28-img.shape[1])//2)+img.shape[1]] += img
                                
                                # plt.imshow(temp)
                                # plt.show()
                                self.number_pred += str(list(range(10))[np.argmax(self.number_detector.predict(np.expand_dims(temp, 0)))])

                        if self.angle < 0:
                            if abs(self.angle) > 90:
                                self.angle = 180 - self.angle
                            else:
                                self.angle = abs(self.angle)
                        else:
                            if self.angle > 90:
                                self.angle  = 180 - self.angle
                        # print(self.angle)
                        
                elif len(approx) > 7:
                    # if list(self.last.values())[0][0] == "circle":
                    #     self.shape_pred = None
                    #     return 0
                    circles = cv2.HoughCircles(cv2.Canny(self.thresh, 0, 100), cv2.HOUGH_GRADIENT, 3, 3, minRadius=min(self.thresh.shape)//3)
                    
                    (x, y), r = cv2.minEnclosingCircle(last_cnts)
                    if circles is not None and r is not None:
                        circles = np.array(circles.reshape(-1, 3))
                        for c in range(len(circles)):
                            circle = circles[c]
                            if abs(circle[0] - x) / ((circle[0] + x)/2) < .1 and abs(circle[1] - y) / ((circle[1] + y)/2) < .1 and abs(circle[2] - r) / ((circle[2] + r)/2) < .1:
                                break
                    else:
                        r = None                                
                    if r is not None:
                        self.shape_pred = None
                        self.res = "circle"
                        self.circle_params = [int(x), int(y), int(r)]

                        x, y, w, h = cv2.boundingRect(last_cnts)
                        # cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 0, 255))
                        img = self.thresh[y:y+h, x:x+w].copy()
                        prob = max(img.shape) / 120
                        img = cv2.resize(img, (int(img.shape[1] / prob), int(img.shape[0] / prob)))
                        temp = np.zeros((128, 128), np.uint8)
                        temp[(128-img.shape[0])//2:((128-img.shape[0])//2)+img.shape[0], (128-img.shape[1])//2:((128-img.shape[1])//2)+img.shape[1]] += img
                        label = ['H', 'L', 'T', 'X']
                        # plt.imshow(temp)
                        # plt.show()
                        
                        
                        self.shape_pred = label[np.argmax(self.shape_detector.predict(np.expand_dims(temp/255, 0)))]
                        
                        cv2.circle(self.image, [self.circle_params[0], self.circle_params[1]], 3, (0, 0, 255)) # daire merkez noktası
                        cv2.circle(self.image, [self.circle_params[0], self.circle_params[1]], self.circle_params[-1], (0, 0, 255)) # daire merkez noktası
        else:
            self.last = {0:[0]}
    def lineFollow(self):
        if self.circle_params is not None:
            self.thresh = cv2.erode(cv2.circle(self.thresh, (self.circle_params[0], self.circle_params[1]), self.circle_params[-1], 0, -1), np.ones((3, 3)))
        cnts, _ = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y = 160, 90
        if len(cnts) != 0:
            last_cnts = cnts[np.argmax([cv2.contourArea(cnt) for cnt in cnts])]

            M = cv2.moments(last_cnts)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                self.lineCenter = [x,y]
        self.lineCenter = [x,y] # else            
        cv2.circle(self.image, (x,y), 3, (255, 0, 0), -1)
        
    # def predictNumber(self):
    #     self.model = 

            
            
odtu = ODTU()
odtu.last = {0:[0]}
odtu.number_detection()
odtu.shape_detection()
files = []

found = {}

for c, file in enumerate(os.listdir("data01")):
    try:
        files.append(int(file[:-4]))
    except:
        print("resimlere eklenmedi :", file[:-4])
    
files.sort()
# files = files[450:650]
s = time.time()
lines = open("data01/line_frames.txt", "r").readlines()
lines = [line.replace("\n", "") for line in lines]
length = len(lines[0])
# ['00147', '00178', '00268', '01185', '01248', '01303', '01583']

# file = 0
# cap = cv2.VideoCapture("dance.mp4")
# while cap.isOpened():
#     ret, image = cap.read()
#     file += 1
for c, file in tqdm(enumerate(files)):
    image = imread("data01/"+ "0"*(length-len(str(file))) + str(file)+".png")

    line = None
    odtu.prepareImage(image)
    odtu.detectShape()
    
    if "0"*(length-len(str(file))) + str(file) in lines:
        # if circle_params is not None
        odtu.lineFollow()
        if odtu.lineCenter[0] - (odtu.image.shape[1]/2) > 0:
            line = 0
            # print("sağ", end=" ")
        else:
            line = 1
            # print("sol", end=" ")
        # follow line
    if line is not None:
        if found.get("line") is None:
            found["line"] = [[file, line]]
        else:
            found["line"].append([file, line])
        #odtu.last = found[-1]
    elif odtu.res is not None:
        if odtu.res =="arrow":
            if found.get("arrow") is None:
                found["arrow"] = [[file, odtu.angle, odtu.number_pred]]
            else:
                found["arrow"].append([file, odtu.angle, odtu.number_pred])
            # print(odtu.number_pred)
        if odtu.res == "circle":
            if found.get("circle") is None:
                found["circle"] = [[file, odtu.circle_params[0], odtu.circle_params[1], odtu.shape_pred]]
            else:
                found["circle"].append([file, odtu.circle_params[0], odtu.circle_params[1], odtu.shape_pred])
            # print(odtu.shape_pred)
    
    # summ = np.hstack((odtu.image, np.stack((odtu.thresh, odtu.thresh, odtu.thresh), axis=-1)))
    # cv2.imshow("frames", summ)
    # print(found)

    # if cv2.waitKey(1) & 0xff == ord(" "):
    #     break


# circle = found["circle"]


circle = np.array([found.get("circle")])[0]
line = np.array([found.get("line")])[0]
arrow = np.array([found.get("arrow")])[0]

if circle is None:
    circle = []

if line is None:
    line = []

if arrow is None:
    arrow = []
f = 0
final = []
# temp = data["circle"][0]
for i in range(len(circle)-1):
    if abs(int(circle[i, 0]) - int(circle[i+1, 0])) > 10 or i==len(circle)-2:
        if i+1 - f > 10:
            most_repeat = np.unique(circle[f:i+1, -1], return_counts=True)[0][np.argmax(np.unique(circle[f:i+1, -1], return_counts=True)[1])]
            result_string = "0"*(length- len(str(circle[f, 0])))+str(circle[f,0]) + "_" + str(circle[f, 1]) + "_" + str(circle[f,2]) + "_" + most_repeat
            if len(final) == 0:
                final.append(result_string)
            elif final[-1][-1] != result_string[-1]:    
                final.append(result_string)
            f = i+1

f = 0
for i in range(len(arrow)-1):
    if abs(int(arrow[i, 0]) - int(arrow[i+1, 0])) > 10 or i==len(arrow)-2:
        if i+1 - f > 10:
            most_repeat = np.unique(arrow[f:i+1, -1], return_counts=True)[0][np.argmax(np.unique(arrow[f:i+1, -1], return_counts=True)[1])]
            result_string = "0"*(length - len(str(arrow[f, 0])))+str(arrow[f,0]) + "_" + str(arrow[f, 1]).split(".")[0] + "_" + most_repeat
            if len(final) == 0:
                final.append(result_string)
            elif final[-1].split("_")[-1] != result_string.split("_")[-1]:    
                final.append(result_string)
            f = i+1

f = 0
for i in range(len(line)):
    result_string = "0"*(length - len(str(line[i, 0])))+str(line[i,0]) + "_" + str(line[i, 1])
    final.append(result_string)

with open("output.txt", "w") as file:
    file.write("\n".join(final))

# import json
# file = open("data.json" , "w")
# json.dump(found, file)
# file.close()

# cv2.destroyAllWindows()