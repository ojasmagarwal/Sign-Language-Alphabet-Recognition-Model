#importing the modules

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#PART 1 - DATA PREPROCESSING

#Preprocessing the training set

train_datagen = ImageDataGenerator(    #image augmentation - we apply transformations to the images to avoid overfitting, we apply transformations like zooming in and zooming out,horizontal flip,rotations etc. we just apply these transformations to the training set
                rescale=1./255,   #feature scaling
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,validation_split=0.2)

training_set= train_datagen.flow_from_directory(  #flow from directory connects the image augmentation tool to the database
            directory='Indian',
            target_size=(64,64), #resizing the images so that pc takes less time
            shuffle=True,
            class_mode='categorical') #if more than 2 then write 'categorical'

#Preprocessing the test set

test_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)
test_set = test_datagen.flow_from_directory(directory='Indian',
            target_size=(64,64),
            shuffle=True,
            class_mode='categorical')

#PART 2 - BUILDING THE CNN

#initialising the cnn

cnn = tf.keras.models.Sequential()

#Step 1 - COnvolution

cnn.add(tf.keras.layers.Conv2D(filters =32,kernel_size=3,activation = 'relu',input_shape=[64,64,3]))   #adds a convolutional layer
#kernel size is the size of the feature detector, input shape is resized to the size we did at data preprocessing and the last parameter is 3 because the images is colored, if the images were black and white the value would be 1

#Step 2 - Pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))

#Adding a second convolutional layer

cnn.add(tf.keras.layers.Conv2D(filters =32,kernel_size=3,activation = 'relu'))  # we remove the input shape here because the cnn already has input shape in the first layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))

#Step 3 - Flattening

cnn.add(tf.keras.layers.Flatten())

#Step 4 - Full Connection

cnn.add(tf.keras.layers.Dense(units = 128,activation='relu'))

#Step 5 - Output Layer

cnn.add(tf.keras.layers.Dense(units = 35,activation='softmax'))

#PART 3 - TRAINING THE CNN

#Compiling the CNN

cnn.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Training the CNN on the training set and evaluating it on the test set

#cnn.fit(x=training_set,validation_data=test_set,epochs = 2)

#Saving the model and reconstructing it

#cnn.save('epoch_2')
reconstructed_cnn = tf.keras.models.load_model('epoch_2')
#reconstructed_cnn.fit(x=training_set,validation_data=test_set)

#STEP 4 - MAKING A SINGLE PREDICTION

from keras.preprocessing import image
from keras.utils import load_img
from keras.utils import img_to_array
test_image = load_img('C:\\Users\\ojasm\\Documents\\ML\\Indian\\D\\5.jpg',target_size=(64,64))
test_image.show()
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = reconstructed_cnn.predict(test_image)
print(np.argmax(result))
label_map=training_set.class_indices 
print(label_map)
