#Import libs
import numpy as np
import cv2
import pymeanshift as pms
from pyimagesearch import imutils
from matplotlib import pyplot as plt
import os

def centre_letter(letter_in):
	for (x,y,w,h) in letter_in:
		centre_x = int((x + (0.5 * w)))
		centre_y = int((y + (0.5 * h)))

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

#mN = int(raw_input('Enter number of minNeighbors: '))
mN = []
zeroth = []
stage = []
q = 0
apple = 0

for a in xrange(7):
	mN.append(10)
	zeroth.append(0)
	stage.append(0)
	

#Run loop using the web camera
while(q <= 5):
	#Read in the image from the webcam
	ret, img = cap.read()

	#Convert the image in to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 80, 100)
	
	#Read in the dimensions of the letter
	A = letter_A.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[0])
	B = letter_B.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[1])
	C = letter_C.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[2])
	D = letter_D.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[3])
	E = letter_E.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[4])
	F = letter_F.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[5])
	G = letter_G.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN[6])
	
	pizza = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
	beer = [A,B,C,D,E,F,G]
	party = zip(pizza, beer)
	
	#Step 1 - determine the lettes		
	i = 0
	while (i<10):
		k = 0
		for letters in beer:
			centre_letter(letters)
			if len(letters) == 0:
				zeroth[k] += 1
			else:
				zeroth[k] += 0
			k += 1
		x = 1	
		i += 1
	
	new_party = []
	
	j = 0
	for p0 in party:
		if zeroth[j] != 10:
			new_party.append(party[j])
			mN[j] += 5
		else:
			mN[j] = mN[j]	
		j += 1
	
	n = 0
	end_party = []
	for p1 in new_party:
		if len(new_party[n][1]) == 1:
			end_party.append(new_party[n][0])
		n += 1
	
	yay = 0
	if len(end_party) == 1:
		print 'Stage %d Letter Found: %s' % (q, end_party[0])
		for b in pizza:
			mN[yay] = 10
			if pizza[yay] == str(end_party[0]):
				apple += 1
				stage[yay] = apple
			else:
				stage[yay] = 0
			yay += 1
		q += 1
		

	
	'''
	if len(stage) == 5:
		for cheese in pizza:
			stage.count(
			print stage
			break
	'''
		
#Close window and turn off the webcam		
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
