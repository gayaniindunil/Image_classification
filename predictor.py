'''
this is the predict class which is written to predict images
using the cifar10_cnn model
'''


import numpy as np
import os
import cv2

import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model('zoo/model/model1.h5')



model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('cnn_test_models/images/flower.jpg')
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.resize(img,(32,32))
img = np.reshape(img,[1,32,32,3])
print "Predicting the image"
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)

datagen.fit(img)
classes = model.predict_classes(img, batch_size=10)

num_predictions = 20

#for predict_index, predicted_y in enumerate(classes):
    #actual_label = 'dog'
    #actual_label = labels['label_names'][np.argmax(y_test[predict_index])]
    #predicted_label = labels['label_names'][np.argmax(predicted_y)]
    #print('Actual Label = %s vs. Predicted Label = %s' % (actual_label,
    #                                                      predicted_label))

print classes
