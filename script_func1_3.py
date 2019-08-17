
import cv2
import os
import csv

import pandas as pd
import numpy as np

def f_segmentacao(indice):
    path_1 = './dir_1'
    path_im1 = str(indice)+'_training.tif'
    img = cv2.imread(os.path.join(path_1,path_im1))

    r = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    threshold=cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 2.)
    ret,f6 = cv2.threshold(threshold,0,250,cv2.THRESH_BINARY)

    kernel = np.ones((2,2))
    morph_img = cv2.morphologyEx(f6, cv2.MORPH_CLOSE, kernel)
    imagem = cv2.bitwise_not(morph_img)
    imagem = cv2.bilateralFilter(imagem,9,55,55)
    imagem = cv2.medianBlur(imagem,5)
    imagem = cv2.bilateralFilter(imagem,7,55,55)
    imagem = cv2.dilate(imagem,kernel,iterations=5)
    imagem = cv2.medianBlur(imagem,5)

    img_final = imagem

    #cv2.moveWindow("1", 40,30)  # Move it to (40,30)
    #cv2.imshow("transformado", imagem)

    path_2 = 'C:/.../projeto_aprmaqu/dir_2'
    path_im2 = str(indice)+'_proc_out_train.png'
    cv2.imwrite(os.path.join(path_2, path_im2), img_final)
    
for indice in range (21,41):
    f_segmentacao(indice)
    
def write_imgs_csv():
    df = pd.DataFrame()
    
    for indice in range(21,41):
        path_2 = 'CC:/.../projeto_aprmaqu/dir_2'
        path_im2 = str(indice)+'_proc_out_train.png'
        img = cv2.imread(os.path.join(path_2,path_im2))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape_img = img.shape
        array_img = img.reshape(1,shape_img[0]*shape_img[1])
        img_series = pd.Series([array_img])
        pd.concat([df,img_series], axis=1)

    #df.head()
    path_3 = 'C:/.../projeto_aprmaqu//dir_3'
    path_csv = 'imgs_segmentadas.csv'
    df.to_csv(os.path.join(path_3,path_csv), encoding='utf-8', index=False)

write_imgs_csv()
