# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:34:50 2022

@author: Michał Odziemkowski i Hubert Parada
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#skanowanie filmu w poszukiwaniu obiektów
def detector(video, height, width, counter, objects_list):
    i = 0
    change_threshold1 = 3 #wartosć dolnego progu zczytywania barw
    change_threshold2 = 255 #wartosć górnego progu zczytywania barw
    lower_threshold = np.array([change_threshold1,change_threshold1,change_threshold1])
    upper_threshold = np.array([change_threshold2,change_threshold2,change_threshold2])
    
    _, previous = video.read() #wczytywanie poprzedniej klatki
    cut_previous = previous[0:height, 0:width] #docinanie klatki na wymiar największego obiektu
    cut_previous = cv2.GaussianBlur(cut_previous,(3,3),0) #implementacja rozmycia w celu eliminacji szumów
    cut_previous_scan = cut_previous[height-3:height-2, 0:width] #przygotowanie obszaru pod skanowanie w poszukiwaniu obiektu
    
    while(True):
        counter += 1 #zwiększanie zmiennej okrelającej obecną klatkę
        _, frame = video.read() #wczytywanie obecnej klatki
        if counter > 26: #rozpoczęcie analizy wideo po przejsciu napisów początkowych
            cut_frame_export = frame[0:height, 0:width]
            cut_frame = cv2.GaussianBlur(cut_frame_export,(3,3),0)
            cut_frame_scan = cut_frame[height-3:height-2, 0:width]
            difference = cv2.absdiff(cut_frame_scan,cut_previous_scan)#analiza różnicy pomiędzy obecną i poprzednią klatką
            mask = cv2.inRange(difference,lower_threshold,upper_threshold)
            changes = (mask > 0).sum()#okreslenie wartosci zmian
                        
            i += 1
            if changes > 3 and counter > 27 and i >= 0: #warunek dla obiektu w obszarze
                i += 1
                objects_list[check_area(cut_frame_export)] += 1 #przekazanie info o obiekcie do słownika
                check_colour(cut_frame_export)
                info_board(frame, objects_list)
                i = -25 #wyłączenie analizy na czas 25 klatek od pojawienia się obiektu w obszarze
                
            cut_previous_scan = cut_frame_scan
            cv2.imshow('original',frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if counter == 1999:
            break
        
    video.release()
    cv2.destroyAllWindows()
    # cv2.destroyWindow('original')

#badanie wielkosci i kształtu obiektów
def check_area(frame):
    b = cv2.inRange(frame,(0,0,0),(350,55,100))
    edges = cv2.Canny(b, 10, 100) #Znajdywanie narożników obiektóW
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, 0, (0,230,255), 1)
    colour = check_colour(frame) #przekazanie koloru
    
    for cnt in contours:
        area = cv2.contourArea(cnt)

    if area > 550 and area < 730:
        return f"big, {colour} circle"
        
    if area > 230 and area < 365:
        return f"medium, {colour} circle"
    
    if area > 125 and area < 230:
        return f"small, {colour} circle"
        
    if area > 730 and area < 900:
        return f"big, {colour} square"
        
    if area > 365 and area < 550:
        return f"medium, {colour} square"
    
    if area > 105 and area < 125:
        return f"small, {colour} square"

#analiza koloru obiektu
def check_colour(saved_frame):
    saved_frame = cv2.GaussianBlur(saved_frame, (3,3), 0)
    saved_frame_rgb = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2RGB)    
    
    # wyznaczanie wektora największego koloru w danej klatce
    colour = np.amax(np.amax(saved_frame_rgb,axis=0), axis=0)
    
    if colour[2] > 25 and colour[2] < 40:
        return "red"
        
    if colour[0] > 120 and colour[0] < 170:
        return "blue"
        
    if colour[2] > 189 and colour[2] < 198:
        return "pink"

#tworzenie i aktualizacja wykrytych obiektóW oraz eksportowanie raportu końcowego w formacie png
def info_board(frame, objects_list):
    image = cv2.imread("PA_4_ref.png")
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0,0,0)
    thickness = 1
    ox = -5 #offset x
    oy = 4 #offset y
    bpc = (32+ox, 34+oy)
    mpc = (119+ox, 31+oy)
    spc = (203+ox, 25+oy)
    bbc = (32+ox, 114+oy)
    mbc = (119+ox, 111+oy)
    sbc = (203+ox, 105+oy)
    brc = (32+ox, 194+oy)
    mrc = (119+ox, 190+oy)
    src = (203+ox, 185+oy)
    bps = (302+ox, 34+oy)
    mps = (398+ox, 41+oy)
    sps = (485+ox, 44+oy)
    bbs = (302+ox, 114+oy)
    mbs = (398+ox, 121+oy)
    sbs = (485+ox, 124+oy)
    brs = (302+ox, 194+oy)
    mrs = (398+ox, 202+oy)
    srs = (485+ox, 204+oy)
    for key, value in objects_list.items():
        if key == "big, red circle":
            text = f"{value}"
            image = cv2.putText(image, text, brc, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "medium, red circle":
            text = f"{value}"
            image = cv2.putText(image, text, mrc, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "small, red circle":
            text = f"{value}"
            image = cv2.putText(image, text, src, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "big, blue circle":
            text = f"{value}"
            image = cv2.putText(image, text, bbc, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "medium, blue circle":
            text = f"{value}"
            image = cv2.putText(image, text, mbc, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "small, blue circle":
            text = f"{value}"
            image = cv2.putText(image, text, sbc, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "big, pink circle":
            text = f"{value}"
            image = cv2.putText(image, text, bpc, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "medium, pink circle":
            text = f"{value}"
            image = cv2.putText(image, text, mpc, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "small, pink circle":
            text = f"{value}"
            image = cv2.putText(image, text, spc, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "big, red square":
            text = f"{value}"
            image = cv2.putText(image, text, brs, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "medium, red square":
            text = f"{value}"
            image = cv2.putText(image, text, mrs, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "small, red square":
            text = f"{value}"
            image = cv2.putText(image, text, srs, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "big, blue square":
            text = f"{value}"
            image = cv2.putText(image, text, bbs, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "medium, blue square":
            text = f"{value}"
            image = cv2.putText(image, text, mbs, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "small, blue square":
            text = f"{value}"
            image = cv2.putText(image, text, sbs, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "big, pink square":
            text = f"{value}"
            image = cv2.putText(image, text, bps, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "medium, pink square":
            text = f"{value}"
            image = cv2.putText(image, text, mps, font, fontScale, color, thickness, cv2.LINE_AA)
        if key == "small, pink square":
            text = f"{value}"
            image = cv2.putText(image, text, sps, font, fontScale, color, thickness, cv2.LINE_AA)
    sum_text = f"Total objects: {sum(objects_list.values())}"
    image = cv2.putText(image, sum_text, (12, 270), font, fontScale+0.5, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imshow("Text", image) #eksport do okienka
    cv2.imwrite("Summary.png", image) #eksport do pliku
    show_image(image) #eksport do "Plots"

def show_image(image, title = "", axis = False, openCV = True, colmap = 'gray'):
    if not(axis):
        plt.axis("off") 
    if image.ndim == 2:
        plt.imshow(image,cmap = colmap,vmin=0, vmax=255)
    else:
        if openCV:
            plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),vmin=0, vmax=255)
        else:
            plt.imshow(image,interpolation = 'none',vmin=0, vmax=255)
    plt.title(title)