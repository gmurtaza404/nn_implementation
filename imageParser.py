import os
import sys
import numpy as np
import scipy.misc
import random
from PIL import Image

def ImageParser():
    mainDirectory =  os.getcwd()
    files = [f for f in os.listdir('.') if not os.path.isfile(f)]
    files.remove('models')
    print files
    print "PARSING TEST DATA"
    os.chdir(files[1])
    tempDirectory = os.getcwd()
    numbers = os.listdir('.')
    ImagesList = []
    tempDirectory = os.getcwd()
    FinalList = []
    for digits in numbers:
        os.chdir(digits)
        imagelabel = digits
        print imagelabel
        Images = [f for f in os.listdir('.') if os.path.isfile(f)]
        for image in Images:
            img = Image.open(image).convert('P')
            readImage=  np.array(img);
            readImage = readImage - 255
            readImage = np.absolute(readImage)
            FinalList.append(readImage)
            #print readImage
            #print "\n \n \n \n \n "
        os.chdir(tempDirectory)
    os.chdir(mainDirectory)
    print len(FinalList)

    os.chdir(files[0])
    tempDirectory = os.getcwd()
    numbers = os.listdir('.')
    ImagesList = []
    tempDirectory = os.getcwd()
    FinalListTest = []
    for digits in numbers:
        os.chdir(digits)
        imagelabel = digits
        print imagelabel
        Images = [f for f in os.listdir('.') if os.path.isfile(f)]
        for image in Images:
            img = Image.open(image).convert('P')
            readImage=  np.array(img);
            readImage = readImage - 255
            readImage = np.absolute(readImage)
            FinalListTest.append(readImage)
            #print readImage
            #print "\n \n \n \n \n "
        os.chdir(tempDirectory)
    os.chdir(mainDirectory)
    print len(FinalListTest)

    return [FinalList,FinalList]


if __name__ == "__ImageParser__":
 	ImageParser();
