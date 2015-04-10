The file letter_detect.py uses classifiers, products from the Haar classifier training.
These classifiers are designed to detect a letter.

The letter it is trained to detect is a white letter inside a red box.
In order to detect the letter the function detectMultiScale is used.

cv2.CascadeClassifier.detectMultiScale(image, rejectLevels, levelWeights[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize[, outputRejectLevels]]]]]]) → objects
Parameters:	
  cascade – Haar classifier cascade (OpenCV 1.x API only). It can be loaded from XML or YAML file using Load(). When the cascade is not needed anymore, release it using cvReleaseHaarClassifierCascade(&cascade).
  image – Matrix of the type CV_8U containing an image where objects are detected.
  objects – Vector of rectangles where each rectangle contains the detected object.
  scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
  minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
  flags – Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
  minSize – Minimum possible object size. Objects smaller than that are ignored.
  maxSize – Maximum possible object size. Objects larger than that are ignored.
  
minNeighbors is the parameter that is increased in order to detemine the final accuracy of the letter being recognised.

The problem: The code does always produce the right result. If I hold the letter G up to the camera it will say a B or C is
  detected. Is there a way on increasing the accuracy based on my code or will there have to be a different approach to detect
  the right letter?

letter_detect.py:

Lines 18 - 24: Reads in the classifier files

Lines 27 - 31: Creates variables

Lines 33 - 36: Builds empty arrays - mN is set to 10 and the others are set to 0.

Lines 40 - 105: While loop that increases the value of minNeighbors to finally determine the letter

  Line 42: Reads in the webcam and saves it 'img'
  
  Lines 45 - 47: Converts 'img' into a gray image (this makes it easier to detect the letter later on)
  
  Lines 50 - 56: The parameters are set for detectMultiScale, where minNeighbours are read in as an array
  
  Lines 58 - 60: An array is bulit for each letter and zipped together
  
  Lines 62 - 73: Here the loop will run 10 times: if, for example the letter A is not detected, then it return a score of 1.
    If the letter, say C, is detected then it will be return the score of 0. The loop will build up the array to determine
    what potential letters are being detected.
    
  Line 75: A new Array is created
  
  Lines 77 - 84: For each letter that scored 10 the value of mN is kept the same. If the value does not equal 10 then the
    number of minNeighbours is increased by 5
    
  Lines 86 - 91: For each letter that did not score 10 is added to new_party. If the length of new_party is equal to 1 this 
    is added to end_party (the final confirmation of the letter detected)
    
  Lines 94 - 104: When end_party contains 1 letter the letter detected is printed on the screen. the variable q is increased
    by 1 when this is complete and the variable mN is reset to 10. This will continute for 5 'passes'.
    
    Lines 96 - 103: The idea was to build an array to output a probability of what the letter could be, but I got stuck.
