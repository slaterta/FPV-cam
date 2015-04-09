#Import libs
import numpy as np
import cv2
import pymeanshift as pms
from pyimagesearch import imutils
from matplotlib import pyplot as plt
'''
def index_letter(letter):
	
	return index
	
def img_process(img, rect):

	
			

			orig = img.copy()
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.imshow("orig", img)
			
			
			
			cv2.rectangle(mask,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.imshow("Masked", cv2.bitwise_and(orig, orig, mask))


	# Initiate FAST object with default values
	fast = cv2.FastFeatureDetector()

	# find and draw the keypoints
	kp = fast.detect(edged,None)
	img2 = cv2.drawKeypoints(edged, kp, color=(255,0,0))
	
	#fast_false
	fast.setBool('nonmaxSuppression',0)
	kp = fast.detect(edged,None)
	img3 = cv2.drawKeypoints(edged, kp, color=(255,0,0))
	'''
	#plt.imshow(img),plt.colorbar(),plt.show()
	#if len(mask) != 0:
	
		
		
	
	
	#cv2.imshow('img', img3)
	


#Turn on the camera
cap = cv2.VideoCapture(0)

#Read in the the Classifier for the letter
letter_A = cv2.CascadeClassifier('A.xml')
letter_B = cv2.CascadeClassifier('B.xml')
letter_C = cv2.CascadeClassifier('C.xml')
letter_D = cv2.CascadeClassifier('D.xml')
letter_E = cv2.CascadeClassifier('E.xml')

mN = 10


A_count = 0
B_count = 0
C_count = 0
D_count = 0
E_count = 0

zero_count = [A_count, B_count, C_count, D_count, E_count]

i = 0
#Run loop using the web camera
while(True):
	x = 0
	
	#Read in the image from the webcam
	ret, img = cap.read()

	#Convert the image in to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 20, 100)

	#Read in the dimensions of the letter
	A = letter_A.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN)
	B = letter_B.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN)
	C = letter_C.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN)
	D = letter_D.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN)
	E = letter_E.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=mN)

	pizza = ['A', 'B', 'C',' D', 'E']
	chips = [A,B,C,D,E]
	
	zeroth = []
	

	
	while (i<1):
		
		j = 0
		k = 0
		
		for letters in chips:
		
			#mask = []
			#ratio = img.shape[0] / 300.0
			#orig = img.copy()
			#img = imutils.resize(img, height = 300)

			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (5, 5), 0)
			edged = cv2.Canny(gray, 20, 100)
	
	
	
			if len(letters) != 0:
				for (x,y,w,h) in letters:
			
					#rect_coor = [(x,y),(x+w,y-h),(x+w,y+h),(x-w,x+h)]
					mask = np.zeros(img.shape[:2], dtype = "uint8")
					
					bgd = np.zeros((1,65),np.float64)
					fgd = np.zeros((1,65),np.float64)
					
					rect_1 = (x,y,w,h)
					cv2.grabCut(img,mask,rect_1,bgd,fgd,5,cv2.GC_INIT_WITH_RECT)
					
					mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
					img = img*mask2[:,:,np.newaxis]
					
					


			if len(letters) == 0:
				zeroth.append(k)
				
			else:
				zeroth.append(0)
				
				
			j += 1
			k += 1
			
		i += 1
	plt.imshow(img),plt.colorbar(),plt.show()
	#print zeroth	
		#zero_count.sort()
		#print zero_count

	#cv2.imshow('img', img)
	#print img


		
#Close window and turn off the webcam		
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
