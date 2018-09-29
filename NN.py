import numpy as np
import random
import sys
import os
import math
import re
import scipy as sp

def main():
    print "STARTING THE NETWORK"
    net = NeuralNetwork([784,30,10],0.3)
    #net.load()
    net.trainNetwork("train.txt", "train-labels.txt")
    net.testNetwork("test.txt","test-labels.txt")
    net.save()


class NeuralNetwork(object):
    def __init__(self,sizeList,alpha):
        self.networkSizeList = sizeList;
        self.networkSize = len(sizeList)
        self.inputList =  np.zeros(self.networkSizeList[0]);
        self.hiddenWeightList = 2*(np.random.rand(self.networkSizeList[1],self.networkSizeList[0]))-1
        self.baisHidden = 0
        self.finalWeightList = 2*np.random.rand(self.networkSizeList[2],self.networkSizeList[1])-1
        self.baisOutput = 0
        self.firstActivations = np.zeros(self.networkSizeList[0]);
        self.secondActivations = np.zeros(self.networkSizeList[1]);
        self.finalOutputList = np.zeros(self.networkSizeList[2]);
        self.learningRate = alpha
        self.outListwoActivation = np.zeros(self.networkSizeList[2]);
        self.hiddenListwoActivation = np.zeros(self.networkSizeList[1]);
        self.bestacc = 0
        self.graphData = []


    def save(self,filename= "model.npz"):
        print "SAVING THE NETWORK"
        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            hiddenWeightList=self.hiddenWeightList,
            finalWeightList = self.finalWeightList,
            networkSizeList = self.networkSizeList,
            networkSize = self.networkSize,
            inputList = self.inputList,
            firstActivations = self.firstActivations,
            secondActivations = self.secondActivations,
            finalOutputList = self.finalOutputList,
            learningRate = self.learningRate,
            hiddenListwoActivation = self.hiddenListwoActivation,
            outListwoActivation = self.outListwoActivation,
            bestacc = self.bestacc,
            graphData = self.graphData
            )

    def load(self,filename='model.npz'):
        print "LOADING THE NETWORK"
        npzMembers = np.load(os.path.join(os.curdir, 'models', filename))
        self.hiddenWeightList = list(npzMembers['hiddenWeightList'])
        self.finalWeightList = list(npzMembers['finalWeightList'])
        self.networkSizeList = list(npzMembers['networkSizeList'])
        self.networkSize = int(npzMembers['networkSize'])
        self.inputList = list(npzMembers['inputList'])
        self.firstActivations = list(npzMembers['firstActivations'])
        self.secondActivations = list(npzMembers['secondActivations'])
        self.finalOutputList = list(npzMembers['finalOutputList'])
        self.learningRate = float(npzMembers['learningRate'])
        self.hiddenListwoActivation = list(npzMembers['hiddenListwoActivation'])
        self.outListwoActivation = list(npzMembers['outListwoActivation'])
        self.bestacc = int(npzMembers['bestacc'])
        self.graphData = list(npzMembers['graphData'])

    def forwardProp(self,inputtext):
        self.firstActivations = inputtext;
        for i in range(0,len(self.secondActivations)):
            tempValue = self.hiddenWeightList[i].dot(inputtext) + self.baisHidden
            self.hiddenListwoActivation[i] = tempValue
            self.secondActivations[i] =self.sigmoidFunction(tempValue)
        for j in range(0,len(self.finalOutputList)):
            tempValue = self.finalWeightList[j].dot(self.secondActivations) + self.baisOutput
            self.outListwoActivation[j] = tempValue;
            self.finalOutputList[j] = self.sigmoidFunction(tempValue)
        return np.argmax(self.finalOutputList)

    def backProp(self,inputtext,answer):
        errorOutput = np.zeros((self.networkSizeList[2],self.networkSizeList[1]))
        errorHidden = np.zeros((self.networkSizeList[1],self.networkSizeList[0]))
        for i in range(0,self.networkSizeList[2]):
            for j in range(0,self.networkSizeList[1]):
                if i == answer:
                    target = 0.99
                else:
                    target = 0.01
                errorOutput[i][j] = -1*(target - self.finalOutputList[i])* self.finalOutputList[i]*(1 -  self.finalOutputList[i])*self.secondActivations[j]
        # for the next layer
        for j in range(0,self.networkSizeList[1]):
            errorTemp =0
            for i in range(0,self.networkSizeList[2]):
                if i == answer:
                    target = 0.99
                else:
                    target = 0.01
                errorTemp = errorTemp + -1*(target-self.finalOutputList[i])*self.finalOutputList[i]*(1 - self.finalOutputList[i])*self.finalWeightList[i][j]

            for k in range(0,self.networkSizeList[0]):
                errorHidden[j][k] = errorTemp*self.secondActivations[j]*(1-self.secondActivations[j])*self.firstActivations[k]

        errorOutput = self.learningRate * errorOutput
        errorHidden = self.learningRate * errorHidden
        self.hiddenWeightList = np.subtract(self.hiddenWeightList,errorHidden)
        self.finalWeightList = np.subtract(self.finalWeightList,errorOutput)

    def sigmoidFunction(self,value):
        if value < 0:
            return 1.0-(1/(1+ math.exp(value)))
        else:
            return (1.0/(1.0 + math.exp(-value)));




    def trainNetwork(self,filename1,filename2):
        print "LOADING FILES"
        images = parseimageFile(filename1)
        imageLabels = parseLabelsFile(filename2)
        testImages = parseimageFile("test.txt")
        testImageLabel = parseLabelsFile("test-labels.txt")

        graphstuff= [0,60000]
        print "\n    TRAINING NETWORK WITH ALPHA = 0.3"
        for k in range(2):
            for i in range(len(images)):
                x  = self.forwardProp(images[i])
                self.backProp(images[i],imageLabels[i])
                if i % 3000 == 0:
                    print i
                    count = 0
                    randomIndexes = np.random.randint(10, size=(3000))
                    for indexes in range(len(randomIndexes)):
                        x = self.forwardProp(testImages[randomIndexes[indexes]])
                        if x == testImageLabel[randomIndexes[indexes]]:
                            count +=1
                    #print count
                    self.graphData.append((i+graphstuff[k],count/3000.0))
                    if count > self.bestacc:
                        self.bestacc = count
                        self.save()

        self.save()
        graphFile = open('Alpha=0.3.txt', 'w')
        for x , y in self.graphData:
            print x , y
            graphFile.write(str(x))
            graphFile.write('\t')
            graphFile.write(str(y))
            graphFile.write('\n')

    def testNetwork(self,filename1,filename2):
        print "LOADING FILES"
        images = parseimageFile(filename1)
        imageLabels = parseLabelsFile(filename2)
        count = 0
        print "\n TESTING NETWORK"

        for i in range(len(images)):
            x  = self.forwardProp(images[i])
            if x  == imageLabels[i]:
                count = count+1
        #        print "True"
        #    else:
        #        print "False"

        print count, "/ " , 10000




def parseimageFile(filename):
    infile = open(filename,'r')
    returnList = []
    line = ""
    templist = []
    for x in infile:
        if x.find('[') != -1:
            line = ""
        line = line + x.strip('\n')
        if x.find(']') != -1:
            tempList = re.split('[ ]+|\]', line)
            tempList.pop(0)
            tempList.pop(len(tempList)-1)
            imageTemp = map(float,tempList)
            image = map(lambda x: x/255, imageTemp)
            mean = np.mean(image)
            std = np.std(image)
            image = map(lambda x: (x-mean)*std, image)
            #print image
            """
            for i in range(len(line)):
                if line[i] == ' ':
                    continue
                else:
                    tempstring = line[i]
                    while(line[i] != ' ' and i < len(line)):
                        if line[i+1] != ' ':
                            tempstring = tempstring + line[i+1]
                            i = i+1

                    templist.append(int(tempstring))
            """
            returnList.append(image)

    return returnList;

def parseLabelsFile(filename):
    infile = open(filename,'r')
    returnList = []
    for x in infile:
        returnList.append(int(x))
    return returnList


if __name__ == "__main__":
    main()
