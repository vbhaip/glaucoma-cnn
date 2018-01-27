import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import csv
from keras.applications.vgg16 import VGG16
model = VGG16()

width = 224
height = 224
'''
img = load_img("rim-flow-data/train/glaucoma/G-1-L.jpg")
x = img_to_array(img) #numpy array
x = x.reshape((1,) + x.shape) #adds on dimension for keras

print(x.shape)'''

model = Sequential()
model.add(Conv2D(4, (4,4), input_shape=(width,height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(8, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.load_weights("sixth_try.h5")

#model = VGG16(weights='imagenet', include_top=True)
#x = Dense(2, activation='softmax', name='predictions')(model.layers[-2].output)

#Then create the corresponding model
#model = Model(input=model.input, output=x)
#model.summary()

opt = optimizers.SGD(lr=1, momentum=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

traindatagen = ImageDataGenerator(
        rescale=1./255,
        samplewise_std_normalization=True)

testdatagen = ImageDataGenerator(
        rescale=1./255,
        samplewise_std_normalization=True)

train_generator = traindatagen.flow_from_directory(
        'rim-flow-data/train',  # this is the target directory
        target_size=(width, height),
        batch_size=5,
        color_mode='rgb')

validation_generator = testdatagen.flow_from_directory(
        'rim-flow-data/validation',
        target_size=(width, height),
        color_mode='rgb')


model.fit_generator(
        train_generator,
        epochs=40,
        validation_data=validation_generator)


model.save_weights('seventh_try.h5')

#fifth_try - 61.29%
#sixth_try - 67%
