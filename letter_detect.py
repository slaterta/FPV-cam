#Import libs
import numpy as np
import cv2
import pymeanshift as pms
from pyimagesearch import imutils
from matplotlib import pyplot as plt
import os
'''
def centre_letter(letter_in):
	for (x,y,w,h) in letter_in:
		centre_x = int((x + (0.5 * w)))
		centre_y = int((y + (0.5 * h)))
	
	return
'''
#Turn on the camera
cap = cv2.VideoCapture(0)	

#Read in the the Classifier for the letter
letter_A = cv2.CascadeClassifier('A.xml')
letter_B = cv2.CascadeClassifier('B.xml')
letter_C = cv2.CascadeClassifier('C.xml')
letter_D = cv2.CascadeClassifier('D.xml')
letter_E = cv2.CascadeClassifier('E.xml')
letter_F = cv2.CascadeClassifier('F.xml')
letter_G = cv2.CascadeClassifier('G.xml')
letter_H = cv2.CascadeClassifier('H.xml')
letter_I = cv2.CascadeClassifier('I.xml')
'''
letter_J = cv2.CascadeClassifier('J.xml')
letter_K = cv2.CascadeClassifier('K.xml')
letter_L = cv2.CascadeClassifier('L.xml')
letter_M = cv2.CascadeClassifier('M.xml')
letter_N = cv2.CascadeClassifier('N.xml')
letter_O = cv2.CascadeClassifier('O.xml')
letter_P = cv2.CascadeClassifier('P.xml')
letter_Q = cv2.CascadeClassifier('Q.xml')
letter_R = cv2.CascadeClassifier('R.xml')
letter_S = cv2.CascadeClassifier('S.xml')
letter_T = cv2.CascadeClassifier('T.xml')
letter_U = cv2.CascadeClassifier('U.xml')
letter_V = cv2.CascadeClassifier('V.xml')
letter_W = cv2.CascadeClassifier('W.xml')
letter_X = cv2.CascadeClassifier('X.xml')
letter_Y = cv2.CascadeClassifier('Y.xml')
letter_Z = cv2.CascadeClassifier('Z.xml')
'''

#mN = int(raw_input('Enter number of minNeighbors: '))
minNeighbors_Array = []
detection_score = []
#stage = []
q = 0


for a in xrange(9):
	minNeighbors_Array.append(10)
	detection_score.append(0)
	#stage.append(0)
	
delta = 0
#Run loop using the web camera
while(q <= 5):
	#Read in the image from the webcam
	ret, img = cap.read()

	#Convert the image in to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 80, 100)
	
	#Read in the dimensions of the letter
	A = letter_A.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=minNeighbors_Array[0])
	B = letter_B.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=minNeighbors_Array[1])
	C = letter_C.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=minNeighbors_Array[2])
	D = letter_D.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=minNeighbors_Array[3])
	E = letter_E.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=minNeighbors_Array[4])
	F = letter_F.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=minNeighbors_Array[5])
	G = letter_G.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=minNeighbors_Array[6])
	H = letter_H.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=minNeighbors_Array[7])
	I = letter_I.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=minNeighbors_Array[8])
	'''
	J = letter_J.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[9])
	K = letter_K.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[10])
	L = letter_L.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[11])
	M = letter_M.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[12])
	N = letter_N.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[13])
	O = letter_O.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[14])
	P = letter_P.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[15])
	Q = letter_Q.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[16])
	R = letter_R.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[17])
	S = letter_S.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[18])
	T = letter_T.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[19])
	U = letter_U.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[20])
	V = letter_V.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[21])
	W = letter_W.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[22])
	X = letter_X.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[23])
	Y = letter_Y.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[24])
	Z = letter_Z.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[25])
	'''
	letter_string = ['A','B','C','D','E','F','G','H','I'] #,'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
	letter_detectMultiScale = [A,B,C,D,E,F,G,H,I] 	#,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z]
	letter_zip = zip(letter_string, letter_detectMultiScale)
	#Step 1 - determine the lettes		
	i = 0
			
	k = 0
	for letters in letter_detectMultiScale:
		#centre_letter(letters)
		if len(letters) == 0:
			detection_score[k] += 1
		else:
			detection_score[k] += 0
	

	detected_letters = []
	
	j = 0
	for p0 in letter_zip:
		if detection_score[j] != 1:
			detected_letters.append(letter_zip[j])
			minNeighbors_Array[j] += 5
		else:
			minNeighbors_Array[j] = minNeighbors_Array[j]	
		j += 1
	
	n = 0
	final_detection = []
	for p1 in detected_letters:
		if len(detected_letters[n][1]) == 1:
			final_detection.append(detected_letters[n][0])
		n += 1
	
	if len(final_detection) == 1:
		print 'Stage %d Letter Found: %s' % (q, final_detection[0])
		print mN
		#for b in :
			#mN[yay] = 10
		q += 1
		
#Close window and turn off the webcam		
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
