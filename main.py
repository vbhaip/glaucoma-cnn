import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import csv

width = 356*4
height = 536*4
'''
img = load_img("rim-flow-data/train/glaucoma/G-1-L.jpg")
x = img_to_array(img) #numpy array
x = x.reshape((1,) + x.shape) #adds on dimension for keras

print(x.shape)'''

model = Sequential()
model.add(Conv2D(16, (20,20), input_shape=(width,height,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(3))
model.add(Activation('softmax'))

#model.load_weights("third_try.h5")
opt = optimizers.SGD(lr=0.01, momentum=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])

traindatagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        samplewise_std_normalization=True)

testdatagen = ImageDataGenerator(
        rescale=1./255,
        samplewise_std_normalization=True)

train_generator = traindatagen.flow_from_directory(
        'rim-flow-data/train',  # this is the target directory
        target_size=(width, height),
        batch_size=10,
        color_mode='grayscale')

validation_generator = testdatagen.flow_from_directory(
        'rim-flow-data/validation',
        target_size=(width, height),
        batch_size=10,
        color_mode='grayscale')

model.fit_generator(
        train_generator,
        epochs=10,
        validation_data=validation_generator)


model.save_weights('third_try.h5')
