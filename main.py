# File        :   main.py (SVM Generator)
# Version     :   1.1.1
# Description :   Scrip that trains, tests and generates a SVM-based per-letter
#                 model using drawn samples. For use with "Android Watch".           :
# Date:       :   Jan 17, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2
import os


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Writes a png image to disk:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Pre-processes the data base:
def prepareDataset(datasetData, datasetPath, mode, verbose):
    (totalClasses, totalSamples, cellWidth, cellHeight) = datasetData
    sampleMatrix = np.empty((totalClasses, totalSamples, cellHeight, cellWidth), dtype=np.uint8)

    if verbose:
        print("prepareDataset>> Preparing dataset for: " + mode)

    classCounter = 0

    for currentDirectory in os.listdir(datasetPath):

        # initial sample counter:
        sampleCounter = 0

        # Get the directory on the current path:
        currentPath = os.path.join(datasetPath, currentDirectory)

        # Get the images on the current path:
        for currentImage in os.listdir(currentPath):

            # create path and read image:
            imagePath = os.path.join(currentPath, currentImage)

            # Read image as grayscale:
            inputImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

            # Invert image:
            # 0 (Black) - Noise, 255 (White) - Shape to analyze
            inputImage = 255 - inputImage

            if verbose:
                showImage("Image: " + str(sampleCounter), inputImage)

            # Set the resizing parameters:
            (imageHeight, imageWidth) = inputImage.shape[:2]

            # Get the resized parameters:
            aspectRatio = imageHeight / imageWidth
            rescaledWidth = cellWidth
            rescaledHeight = int(rescaledWidth * aspectRatio)

            # Resize, if necessary:
            if (imageHeight != rescaledWidth) or (imageWidth != rescaledWidth):
                if verbose:
                    print("prepareDataset>> Resizing sample from: " + str(imageWidth) + " x " + str(imageHeight)
                          + " to: " + str(rescaledWidth) + " x " + str(rescaledHeight))
                # Set new size:
                newSize = (rescaledWidth, rescaledHeight)
                # resize image
                inputImage = cv2.resize(inputImage, newSize, interpolation=cv2.INTER_NEAREST)

            if verbose:
                showImage("Image Resized", inputImage)

            # Set kernel (structuring element) size:
            kernelSize = (3, 3)
            opIterations = 2
            morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)

            # Perform Dilate:
            inputImage = cv2.morphologyEx(inputImage, cv2.MORPH_DILATE, morphKernel, None, None, opIterations,
                                          cv2.BORDER_REFLECT101)

            if verbose:
                showImage("Image Morphed", inputImage)

            # Store in array:
            sampleMatrix[classCounter, sampleCounter] = inputImage

            sampleCounter = sampleCounter + 1
            print("Class: " + str(classCounter) + " Sample: " + str(sampleCounter))

        classCounter = classCounter + 1

    (totalRows, totalCols) = sampleMatrix.shape[:2]
    print("Created matrix with total cols: " + str(totalCols) + " rows: " + str(totalRows) + " for: " + mode)
    return sampleMatrix


# samples path:
platform = "android"
rootDir = "D:"
path = os.path.join(rootDir, "opencvImages", "androidWatch", "samples", platform)

# train and test dirs:
trainDir = "train"
testDir = "test"

# Debug Flags
showImages = True
verbose = False

# Model output directory:
modelPath = os.path.join(rootDir, "opencvImages", "androidWatch", "model", platform)

# * the class dictionary:
classDictionary = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}

# SVM model flags:
saveModel = True
loadModel = False

# Data set info:
totalClasses = len(classDictionary)
# * Number of samples that the script reads for training and testing
trainSamples = 10
testSamples = 8

# Processing image size:
cellHeight = 100
cellWidth = 100

# Train/Test tuples of data info:
trainData = (totalClasses, trainSamples, cellWidth, cellHeight)
testData = (totalClasses, testSamples, cellWidth, cellHeight)

# Training:
trainPath = os.path.join(path, trainDir)
# Get the training matrix:
trainMatrix = prepareDataset(trainData, trainPath, "Training", verbose)

# Test:
testPath = os.path.join(path, testDir)
# Get the test matrix:
testMatrix = prepareDataset(testData, testPath, "Test", verbose)

# Reshape data to plain matrices:
train = trainMatrix.reshape(-1, cellWidth * cellHeight).astype(np.float32)
# Save a deep copy for testing images:
testImages = testMatrix.copy()
test = testImages.reshape(-1, cellWidth * cellHeight).astype(np.float32)

# Create labels for train and test data
k = np.arange(totalClasses)
train_labels = np.repeat(k, trainSamples)[:, np.newaxis]
test_labels = np.repeat(k, testSamples)[:, np.newaxis]

# Check test images:
(totalTestClasses, totalTestSamples) = testImages.shape[:2]

# Loop through each test image, get its class, dictionary value
# and show it:
for f in range(totalTestClasses):
    for m in range(totalTestSamples):

        # Get current test image:
        currentTestImage = testImages[f][m]

        # Get its class from dictionary class name:
        dictionaryValue = test_labels[testSamples * f + m][0]
        currentClass = classDictionary[dictionaryValue]

        if verbose:
            print("Sample: " + str(f) + "-" + str(m) + ", class: " + currentClass)
            showImage("Test Image", currentTestImage)

print("Running SVM...")

# Check if the SVM model should be loaded via an XML file
# or we must create the SVM model from scratch:
if not loadModel:

    print("Creating SVM from scratch...")

    # Create the SVM:
    SVM = cv2.ml.SVM_create()

    # Android:
    SVM.setKernel(cv2.ml.SVM_LINEAR)  # Sets the SVM kernel, this is a linear kernel
    SVM.setType(cv2.ml.SVM_NU_SVC)  # Sets the SVM type, this is a "Smooth" Classifier
    SVM.setNu(0.01)  # Sets the "smoothness" of the decision boundary, values: [0.0 - 1.0]

    # Windows:
    # SVM.setKernel(cv2.ml.SVM_POLY)        # Sets the SVM kernel, this a polynomial kernel
    # SVM.setType(cv2.ml.SVM_C_SVC)         # Again, a smooth classifier
    # SVM.setDegree(1.56)
    # Sets the polynomial degree, values: [>0.0]
    # SVM.setCoef0(1.5)                     # Sets the polynomial coef, values: [real]
    # SVM.setGamma(5.5)                     # Sets the polynomial parameter gamma, values: [>0.0]
    # SVM.setNu(0.10)                       # Sets the decision boundary smoothness, values: [0.0 - 1.0]

    SVM.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 25, 1.e-01))
    SVM.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    print("SVM parameters set...")

    if saveModel:
        print("Saving SVM XML...")
        modelPath = os.path.join(modelPath, "svmModel.xml")
        SVM.save(modelPath)
        print("Saved SVM XML to: " + modelPath)

else:
    print("Loading SVM XML...")
    modelPath = os.path.join(modelPath, "svmModel.xml")
    SVM = cv2.ml.SVM_load(modelPath)
    print("Loaded SVM XML from: " + modelPath)

# Check if SVM is ready to predict:
svmTrained = SVM.isTrained()
if svmTrained:
    print("SVM has been trained, ready to test. ")
else:
    print("Warning: SVM IS NOT TRAINED!")

# Begin Prediction:
svmResult = SVM.predict(test)[1]

# Show accuracy:
mask = svmResult == test_labels
correct = np.count_nonzero(mask)
print("SVM Accuracy: " + str(correct * 100.0 / svmResult.size) + " %")

# Show each test sample and its classification result:
(h, w) = testImages.shape[:2]
labelIndex = 0

for y in range(h):
    currentArray = testImages[y]
    (h, w) = currentArray.shape[:2]
    for x in range(h):
        currentImage = currentArray[x]

        svmPrediction = svmResult[labelIndex][0]
        svmLabel = classDictionary[svmPrediction]

        dictionaryValue = test_labels[labelIndex][0]
        currentLabel = classDictionary[dictionaryValue]

        writeString = "Label: " + str(labelIndex) + ", " + " Dic Value: " + str(dictionaryValue) + ", " + \
                      ", SVM Pred: " + str(svmPrediction) + ", " + "Ground Truth: " + currentLabel + \
                      ", SVM:" + svmLabel

        labelIndex = labelIndex + 1

        print(writeString)

        if showImages:
            # Put label on classified sample:
            currentImage = cv2.cvtColor(currentImage, cv2.COLOR_GRAY2BGR)

            # Re-size image for displaying results:
            (imageHeight, imageWidth) = currentImage.shape[:2]
            aspectRatio = imageHeight / imageWidth
            rescaledWidth = 300
            rescaledHeight = int(rescaledWidth * aspectRatio)
            newSize = (rescaledWidth, rescaledHeight)

            currentImage = cv2.resize(currentImage, newSize, interpolation=cv2.INTER_NEAREST)

            # Set text parameters:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(currentImage, "SVM: " + str(svmLabel), (3, 50), font, 0.5, (255, 0, 255), 1, cv2.LINE_8)

            # Show the classified image:
            showImage("Test Image", currentImage)
