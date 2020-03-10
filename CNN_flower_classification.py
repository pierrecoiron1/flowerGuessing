#Created by Pierre Coiron

#Import the Machine Learning Libraries
print("Starting Program")
print("Importing Machine Learning Libraries")
from tensorflow.keras import models
from keras.models import Sequential
from keras import layers
from tensorflow.keras.utils import to_categorical
from keras.applications.xception import Xception
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
print("Sucessfully Imported Machine Learning Libraries")

print("Importing Auxiliary Libraries")
import os
import numpy as np
import sys
from PIL import image
from sklearn.preprocessing import LabelEncoder
#sys.modules['Image'] = Image 
import matplotlib as plt
print("Sucessfully Imported Auxiliary Libraries")
print("All Libraries Imported")

print("Referencing original directory")
originalDatasetDir= r"C:\Users\peter\OneDrive\Documents\...Actual Documents\School\Spring 2020\Neural Networks\Projects\Project 5\Sample Photos\flower_photos"
print("Completed original data referencing")

print("Creating Base Directory")
baseDir = 'c:/flowers_small'
#os.mkdir(baseDir)
print("Base Directory Created")

print("Creating Directories for Training")
trainDir=os.path.join(baseDir, 'Train')
#os.mkdir(trainDir)
print("Training Folder Created")
print("Creating Training Data Folders")
trainDaisyDir=os.path.join(trainDir, 'Daisy')
trainDandelionDir=os.path.join(trainDir, 'Dandelion')
trainRosesDir=os.path.join(trainDir, 'Roses')
trainSunflowersDir=os.path.join(trainDir, 'Sunflowers')
trainTulipssDir=os.path.join(trainDir, 'Tulips')
print("Training Folders Completed")
print("")

print("Creating Directories for Validation")
validDir=os.path.join(baseDir, 'Validation')
print("Validation Folder Created")
print("Creating Validation Data Folders")
validDaisyDir=os.path.join(validDir, 'Daisy')
validDandelionDir=os.path.join(validDir, 'Dandelion')
validRosesDir=os.path.join(validDir, 'Roses')
validnSunflowersDir=os.path.join(validDir, 'Sunflowers')
validTulipssDir=os.path.join(validDir, 'Tulips')
print("Validation Folders Completed")
print("")

print("Creating Directories for Testing")
testDir=os.path.join(baseDir, 'Test')
print("Testing Folder Created")
print("Creating Testing Data Folders")
testDaisyDir=os.path.join(testDir, 'Daisy')
testDandelionDir=os.path.join(testDir, 'Dandelion')
testRosesDir=os.path.join(testDir, 'Roses')
testSunflowersDir=os.path.join(testDir, 'Sunflowers')
testTulipssDir=os.path.join(testDir, 'Tulips')
print("Testing Folders Completed")
print("")
#######################################################
#######################################################
print("")#build NN
#######################################################
#######################################################
print("Building Neural Network")
convBase=Xception(include_top=True,
                  weights='imagenet',
                  input_tensor=None,
                  input_shape=None,
                  pooling=None,
                  classes=1000)
convBase.summary()
print("Completed Importing InceptionV3")

print("Scale Images Generator")
dataGen=ImageDataGenerator(rescale=1./255)
batchSize=20
print("Completed Scale Images Generator")

def extractFeatures(directory, sampleCount):
    features=np.zeros(shape=(sampleCount,4,4,512))
    labels=np.zeros(shape=(sampleCount))
    generator=dataGen.flow_from_directory(
            directory,
            target_size=(150,150),
            batch_size=batchSize,
            class_mode='binary')
    i=0
    for inputsBatch, labelsBatch in generator:
        featuresBatch=convBase.predict(inputsBatch)
        features[i * batchSize : (i+1) * batchSize]=featuresBatch
        labels[i * batchSize : (i+1) * batchSize] = labelsBatch
        i+=1
        if i * batchSize >=sampleCount:
            break
    return features, labels

print("Extract Features Functions on Trainin, Validation and Testing")
trainFeatures, trainLabels=extractFeatures(trainDir,500)
validationFeatures, validationLabels=extractFeatures(validDir,100)
testFeatures, testLabels=extractFeatures(testDir,33)
print("Completed Extract Features Functions on Trainin, Validation and Testing")

print("Flattening Sample Matricies")
trainFeatures=np.reshape(trainFeatures, (500, 4*4*512))
validationFeatures=np.reshape(validationFeatures, (100, 4*4*512))
testFeatures=np.reshape(testFeatures, (33, 4*4*512))
print("Completed Flattening Sample Matricies")
        
models=models.Sequential()
models.add(layers.Dense(256,
                       activation='relu',
                       input_dim=4*4*512))
models.add(layers.Dropout(0.5))
models.add(layers.Dense(1, activation='sigmoid'))

models.compile(optimizer=optimizers.RMSprop(lr=2e-5),
               loss='binary_crossentropy',
               metrics=['acc'])

history=models.fit(trainFeatures, trainLabels,
                  epochs=30,
                  batch_Size=batchSize,
                  validation_data=(validationFeatures, validationLabels))

acc=history.history['acc']
valAcc=history.history['val_acc']
loss=history.history['loss']
valLoss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, valLoss, 'bo', label='Validation Loss')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

#######################################################
#######################################################
print("")#completed
#######################################################
#######################################################
print("Program Reached Completion")