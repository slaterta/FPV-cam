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
