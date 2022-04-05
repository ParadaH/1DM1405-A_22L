# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:23:13 2022

@author: Michał Odziemkowski i Hubert Parada
"""
import cv2
from Picture_analyzer_fin import detector

if __name__ == "__main__":

    filename_vid = 'PA_4.avi'
    height = 35
    width = 140
    counter = 0
    
    colour_object_list = {"big, red circle": 0,
                          "medium, red circle": 0,
                          "small, red circle": 0,
                          "big, blue circle": 0,
                          "medium, blue circle": 0,
                          "small, blue circle": 0,
                          "big, pink circle": 0,
                          "medium, pink circle": 0,
                          "small, pink circle": 0,
                          "big, red square" : 0,
                          "medium, red square" : 0,
                          "small, red square" : 0,
                          "big, blue square" : 0,
                          "medium, blue square" : 0,
                          "small, blue square" : 0,
                          "big, pink square" : 0,
                          "medium, pink square" : 0,
                          "small, pink square" : 0,}
    video = cv2.VideoCapture(filename_vid)
    detector(video, height, width, counter, colour_object_list)