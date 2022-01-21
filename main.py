# File        :   main.py (SVM Generator)
# Version     :   1.2.0
# Description :   Scrip that trains, tests and generates a SVM-based per-letter
#                 model using drawn samples. For use with "Android Watch".           :
# Date:       :   Jan 20, 2022
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


# Function that fills corners of a square image:
# ( cv::Mat &inputImage, int fillColor = 255, int fillOffsetX = 10, int fillOffsetY = 10, cv::Scalar fillTolerance = 4 )
def fillCorners(binaryImage, fillColor=255, fillOffsetX=10, fillOffsetY=10):
    # Get image dimensions:
    (imageHeight, imageWidth) = binaryImage.shape[:2]
    # Flood-fill corners:
    for j in range(2):
        # Compute y coordinate:
        fillY = int(imageHeight * j + (-2 * j + 1) * fillOffsetY)
        for i in range(2):
            # Compute x coordinate:
            fillX = int(imageWidth * i + (-2 * i + 1) * fillOffsetX)
            # Flood-fill the image:
            cv2.floodFill(binaryImage, mask=None, seedPoint=(fillX, fillY), newVal=(fillColor))
            # print("X: " + str(fillX) + ", Y: " + str(fillY))
            # showImage("Flood-Fill", binaryImage)

    return binaryImage


# Gets the bounding box of a blob via horizontal and
# Vertical projections, crop the blob and returns it:

def getCharacterBlob(binaryImage, verbose=False):
    # Set number of reductions (dimensions):
    dimensions = 2
    # Store the data of the final bounding boxes here,
    # 4 elements cause the list is [x,y,w,h]
    boundingRect = [None] * 4

    # Reduce the image:
    for i in range(dimensions):
        # Reduce image, first horizontal, then vertical:
        reducedImg = cv2.reduce(binaryImage, i, cv2.REDUCE_MAX)
        # showImage("Reduced Image: " + str(i), reducedImg)

        # Get contours, inspect bounding boxes and
        # get the starting (smallest) X and ending (largest) X

        # Find the contours on the binary image:
        contours, hierarchy = cv2.findContours(reducedImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Create temporal list to store the rectangle data:
        tempRect = []

        # Get the largest contour in the contours list:
        for j, c in enumerate(contours):
            currentRectangle = cv2.boundingRect(c)

            # Get the dimensions of the bounding rect:
            rectX = currentRectangle[0]
            rectY = currentRectangle[1]
            rectWidth = currentRectangle[2]
            rectHeight = currentRectangle[3]
            # print("Dimension: " + str(i) + " x: " + str(rectX) + " y: " + str(rectY) + " w: " + str(
            #    rectWidth) + " h: " + str(rectHeight))

            if i == 0:
                # Horizontal dimension, check Xs:
                tempRect.append(rectX)
                tempRect.append(rectX + rectWidth)
            else:
                # Vertical dimension, check Ys:
                tempRect.append(rectY)
                tempRect.append(rectY + rectHeight)

        # Extract the smallest and largest coordinates:
        # print(tempRect)
        currentMin = min(tempRect)
        currentMax = max(tempRect)
        # print("Dimension: " + str(i) + " Start X: " + str(currentMin) + ", End X: " + str(currentMax))
        # Store into bounding rect list as [x,y,w,h]:
        boundingRect[i] = currentMin
        boundingRect[i + 2] = currentMax - currentMin
        # print(boundingRect)

    if verbose:
        print("getCharacterBlob>> Bounding box computed, dimensions as [x,y,w,h] follow: ")
        print(boundingRect)

    # Check out bounding box:
    if verbose:
        binaryImageColor = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)
        color = (0, 0, 255)
        cv2.rectangle(binaryImageColor, (int(boundingRect[0]), int(boundingRect[1])),
                      (int(boundingRect[0] + boundingRect[2]), int(boundingRect[1] + boundingRect[3])), color, 2)
        showImage("BBox", binaryImageColor)

    # Crop the character blob:
    cropX = boundingRect[0]
    cropY = boundingRect[1]
    cropWidth = boundingRect[2]
    cropHeight = boundingRect[3]

    # Crop the image via Numpy Slicing:
    croppedImage = binaryImage[cropY:cropY + cropHeight, cropX:cropX + cropWidth]

    if verbose:
        print("getCharacterBlob>> Cropped image using bounding box data. ")

    return croppedImage


# Process the image before feeding it to the classifier
def processSample(inputImage, verbose=False):
    # Get height and width:
    (inputImageHeight, inputImageWidth) = inputImage.shape[:2]

    # Fill the corners with canvas image
    # inputImage = fillCorners(inputImage, fillOffsetX=5, fillOffsetY=5, fillColor=(255, 255, 255))
    # showImage("inputImage [FF]", inputImage)

    # Threshold the image, canvas color (192) = black, line color (0) = white
    # _, binaryImage = cv2.threshold(inputImage, 1, 255, cv2.THRESH_BINARY_INV)
    binaryImage = inputImage
    # showImage("Binary Image", binaryImage)

    # Get character bounding box via projections and crop it:
    if verbose:
        print("postProcessSample>> Extracting character blob...")

    characterBlob = getCharacterBlob(binaryImage)
    # showImage("characterBlob", characterBlob)

    # Create target canvas with the smallest original dimension:
    largestDimension = min((inputImageHeight, inputImageWidth))

    if verbose:
        print("postProcessSample>> Largest Dimension: " + str(largestDimension))
        print("postProcessSample>> Creating canvas of: " + str(largestDimension) + " x " + str(largestDimension))

    characterCanvas = np.zeros((largestDimension, largestDimension), np.uint8)
    # showImage("characterCanvas", characterCanvas)

    # Get canvas centroid (it is a square):
    canvasX = 0.5 * largestDimension
    canvasY = canvasX

    # Get character centroid:
    (blobHeight, blobWidth) = characterBlob.shape[:2]

    # Get paste x and y:
    pasteX = int(canvasX - 0.5 * blobWidth)
    pasteY = int(canvasY - 0.5 * blobHeight)

    if verbose:
        print("postProcessSample>> Pasting at X: " + str(pasteX) + " ,Y: " + str(pasteY) + " W: " + str(
            blobWidth) + ", H: " + str(blobHeight))

    # Paste character blob into new canvas:
    characterCanvas[pasteY:pasteY + blobHeight, pasteX:pasteX + blobWidth] = characterBlob
    # Invert image:
    characterCanvas = 255 - characterCanvas
    # showImage("Pasted Image", characterCanvas)

    return characterCanvas


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

            if verbose:
                showImage("Image: " + str(sampleCounter), inputImage)

            # Center blob
            # Output image is binary: 0 (Black) - Noise, 255 (White) - Shape to analyze
            inputImage = processSample(inputImage)

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
classDictionary = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
                   10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
                   20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"}

# SVM model flags:
saveModel = True
loadModel = False

# Data set info:
totalClasses = len(classDictionary)
# * Number of samples that the script reads for training and testing
trainSamples = 40
testSamples = 10

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
    SVM.setNu(0.10)  # Sets the "smoothness" of the decision boundary, values: [0.0 - 1.0]

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
