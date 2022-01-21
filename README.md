# svmGenerator
Generates the **SVM model** for use with androidWatch (is not a watch). The Python script reads a directory with two sub-directories: train and test. 
In each sub-directory the are 26 sub-folders, one for each class. Classes are _"A"_ to _"Z"_.

The script extracts _n_ samples for training and testing the SVM. The final **SVM model** is saved as an external _.xml_ to be read by the **androidWatch app**.
The _lettersDS.zip_ is a small dataset of 50 samples for characters _"A"_ to _"Z"_.
