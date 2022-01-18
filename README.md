# svmGenerator
Generates the SVM model for use with androidWatch (is not a watch). The Python script reads a directory with two sub-directories: train and test. 
In each sub-directory the are 26 sub-folders, one for each class. Classes are "A" to "Z".
The script extracts n samples for training and testing the SVM. The final SVM model is saved as an external .xml to be read by the androidWatch app.
