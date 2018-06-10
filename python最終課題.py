# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 20:26:38 2016

@author: 篤樹
"""

#coding: utf-8
# -*- coding: utf-8 -*-
# 画像からカスケード分類器を用いて顔認識を行うサンプル
 
import cv2
import os
import shutil
import numpy as np
 

#~~~~~~~ここから下の二行を環境に合わせて変更してください。~~~~~~~~~~~~~~~
# サンプル顔認識特徴量ファイル
cascade_path = "C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml"
image_path = "lena.jpg"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

file_src = image_path

img_src = cv2.imread(file_src, 1)

cv2.namedWindow('src')
cv2.namedWindow('src2')

color = (255, 255, 255) #白
 
# 画像の読み込み
image = cv2.imread(image_path)
# グレースケール変換
gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
 
# 分類器を作る作業
cascade = cv2.CascadeClassifier(cascade_path)
 
# 顔認識の実行
facerect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(2, 4))

if len(facerect) > 0:
	path = os.path.splitext(image_path)
	dir_path = path[0] + '_face'
	if os.path.isdir(dir_path):
		shutil.rmtree(dir_path)
	os.mkdir(dir_path)
 
     	dir_path2 = path[0] + '_face2'
	if os.path.isdir(dir_path2):
		shutil.rmtree(dir_path2)
	os.mkdir(dir_path2)


#グレースケールに変換し、img_src2と名前つける
img_src2 = cv2.cvtColor(img_src, cv2.cv.CV_BGR2GRAY)

#画像のheight,widthを取得
if len(img_src.shape) == 3:
    height, width, channnels =img_src.shape[:3]
else:
    height, width = img_src.shape[:2]
    channels = 1

#Bayer型のディザ行列で組織的ディザリングの実行
A=np.array([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]])
B=[]

for i in range(4):
    a=[]
    for j in range(4):
            a.append(16*A[i][j])
    B.append(a)

for y in range(height):
    for x in range(width):
        b=[]
        if img_src2[y,x] < B[y%4][x%4]:
            img_src2.itemset((y, x), 0)
        else:
            img_src2.itemset((y, x), 255)

i = 0;
if len(facerect) > 0:
  # 検出した顔を囲む矩形の作成
    for rect in facerect:
        cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
        x = rect[0]
        y = rect[1]
        width2 = rect[2]
        height2 = rect[3]
        
        #dstは元画像のトリミング
        #ｄｓｔ２はディザリング後の画像のトリミング
        dst = image[y:y+height2, x:x+width2]
        dst2 = img_src2[y:y+height2, x:x+width2]
        new_image_path = dir_path + '/' + str(i) + path[1];
        new_image_path2 = dir_path2 + '/' + str(i) + path[1];
        cv2.imwrite(new_image_path, dst)
        cv2.imwrite(new_image_path2, dst2)
        i += 1
else:
  print("no face")
 
# 認識結果の表示
cv2.imshow('src', img_src) # 入力画像を表示
cv2.imshow('dst', image) # 出力画像を表示
cv2.imshow('src2', img_src2)
#cv2.imwrite(file_dst, img_src2); # 処理結果の保存
cv2.waitKey(0) # キー入力待ち
cv2.destroyAllWindows()
