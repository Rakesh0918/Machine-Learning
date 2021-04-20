import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix

#Loading Training training Images
print('Loading Training Images....')
train_images = np.genfromtxt ('train-images.csv', delimiter="," ,skip_header=0, skip_footer=0)
print(train_images)

#Loading Training Labels
print('\nLoading Training Labels....')
train_labels = np.genfromtxt ('train-labels.csv', delimiter="," ,skip_header=0, skip_footer=0)
print(train_labels)

#Setting Up the Classifier
print('\nSVM Classifier with gamma = 0.1; Kernel = polynomial')
clf = svm.SVC(gamma=0.1, kernel='poly')
clf.fit(train_images,train_labels)

#Loading Testing Data
print('\nLoading Testing Data....')
test_images = np.genfromtxt ('test_images.csv', delimiter="," ,skip_header=0, skip_footer=0)
print(test_images)

#Predicting the test Images
print('\nPredicted values of Test Data...')
prediction = clf.predict(test_images)
print(prediction)

#Loading Test Image Labels
print('\nLoading Test Image Labels...')
test_labels = np.genfromtxt ('test_labels.csv', delimiter="," ,skip_header=0, skip_footer=0)
print(test_labels)

#Calculating the accuracy of predicted Values
print('\nCalculating the accuracy of predicted Values...')
print(accuracy_score(test_labels, prediction))

#Confusion Matrix
print('\nGenerating Confusion Matrix: ')
print(confusion_matrix(test_labels, prediction))

# Plot Confusion Matrix for Test Data
plt.matshow(confusion_matrix(test_labels, prediction), cmap="hot", interpolation="nearest")
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Show the Test Images with Original and Predicted Labels
two_d = (np.reshape(test_images[0], (28, 28)) * 255).astype(np.uint8)
plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[0],clf.predict(test_images)[0]))
plt.imshow(two_d, interpolation='nearest',cmap='gray')
plt.show()

two_d = (np.reshape(test_images[1005], (28, 28)) * 255).astype(np.uint8)
plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[1005],clf.predict(test_images)[1005]))
plt.imshow(two_d, interpolation='nearest',cmap='gray')
plt.show()

two_d = (np.reshape(test_images[2005], (28, 28)) * 255).astype(np.uint8)
plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[2005],clf.predict(test_images)[2005]))
plt.imshow(two_d, interpolation='nearest',cmap='gray')
plt.show()

two_d = (np.reshape(test_images[3005], (28, 28)) * 255).astype(np.uint8)
plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[3005],clf.predict(test_images)[3005]))
plt.imshow(two_d, interpolation='nearest',cmap='gray')
plt.show()

two_d = (np.reshape(test_images[4005], (28, 28)) * 255).astype(np.uint8)
plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[4005],clf.predict(test_images)[4005]))
plt.imshow(two_d, interpolation='nearest',cmap='gray')
plt.show()

two_d = (np.reshape(test_images[5005], (28, 28)) * 255).astype(np.uint8)
plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[5005],clf.predict(test_images)[5005]))
plt.imshow(two_d, interpolation='nearest',cmap='gray')
plt.show()

two_d = (np.reshape(test_images[6005], (28, 28)) * 255).astype(np.uint8)
plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[6005],clf.predict(test_images)[6005]))
plt.imshow(two_d, interpolation='nearest',cmap='gray')
plt.show()

two_d = (np.reshape(test_images[7005], (28, 28)) * 255).astype(np.uint8)
plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[7005],clf.predict(test_images)[7005]))
plt.imshow(two_d, interpolation='nearest',cmap='gray')
plt.show()

two_d = (np.reshape(test_images[8005], (28, 28)) * 255).astype(np.uint8)
plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[8005],clf.predict(test_images)[8005]))
plt.imshow(two_d, interpolation='nearest',cmap='gray')
plt.show()

two_d = (np.reshape(test_images[9005], (28, 28)) * 255).astype(np.uint8)
plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[9005],clf.predict(test_images)[9005]))
plt.imshow(two_d, interpolation='nearest',cmap='gray')
plt.show()

