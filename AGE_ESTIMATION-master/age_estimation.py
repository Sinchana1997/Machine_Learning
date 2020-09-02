# USAGE
# python search_shirts.py --dataset shirts --query queries/query_01.jpg

# import the necessary packages
from __future__ import print_function
from tool.localbinarypatterns import LocalBinaryPatterns
from imutils import paths
import numpy as np
import argparse
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle as cPickle

import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True, help="path to the dataset for training images")
ap.add_argument("-e", "--testing", required=True, help="path to the testing images")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor and initialize the index dictionary
# where the image filename is the key and the features are the value
# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# loop over the training images
for imagePath in paths.list_images(args["training"]):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	# extract the label from the image path, then update the
	# label and data lists
	labels.append(imagePath.split("\\")[-2])
	data.append(hist)

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data),
	np.array(labels), test_size=0.25, random_state=42)

# train a KNN on the data
model = KNeighborsClassifier(n_neighbors=1)
print("[INFO] evaluating k-NN classifier...")
model.fit(trainData, trainLabels)
print("[INFO] evaluating...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))
f = open("classifier.cPickle", "wb")
f.write(cPickle.dumps(model))
f.close()


# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	prediction = model.predict(hist.reshape(1, -1))[0]
	clone = image.copy()
	show_image = cv2.resize(clone, (500, 500)) 

	# display the image and the prediction
	cv2.putText(show_image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)
	cv2.imshow("Image", show_image)
	cv2.waitKey(0)



