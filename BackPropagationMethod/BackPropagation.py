import numpy as np
import random
import math


# Function which load the file
def loadFile(fileName):
    resultList = []
    file = open(fileName, 'r')
    for line in file:
        line = line.rstrip('\n')  # get data set -> "1.0,2.0,3.0"
        sVals = line.split(',')  # transform data set into ["1.0", "2.0", "3.0"]
        fVals = list(map(np.float32, sVals))  # transform data set into [1.0, 2.0, 3.0]
        resultList.append(fVals)  # append data set into array [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    file.close()
    return np.asarray(resultList, dtype=np.float32)  # change data type to float32


# This method only used to show the data matrix when running the project.

def showMatrixPartial(m, numberOfRows, dec, indices):
    fmt = "%." + str(dec) + "f"  # like %.4f
    lastRow = len(m) - 1
    width = len(str(lastRow))
    for i in range(numberOfRows):
        if indices == True:
            print("[", end='')
            print(str(i).rjust(width), end='')
            print("] ", end='')

        for j in range(len(m[i])):
            x = m[i, j]
            if x >= 0.0: print(' ', end='')
            print(fmt % x + '  ', end='')
        print('')
    print(" . . . ")

    if indices == True:
        print("[", end='')
        print(str(lastRow).rjust(width), end='')
        print("] ", end='')
    for j in range(len(m[lastRow])):
        x = m[lastRow, j]
        if x >= 0.0: print(' ', end='')
        print(fmt % x + '  ', end='')
    print('')


class NeuralNetwork:
    def __init__(self, numberOfInput, numberOfHiddenLayer, numberOfOutput, seed):
        self.ni = numberOfInput
        self.nh = numberOfHiddenLayer
        self.no = numberOfOutput

        self.iNodes = np.zeros(shape=[self.ni], dtype=np.float32)
        self.hNodes = np.zeros(shape=[self.nh], dtype=np.float32)
        self.oNodes = np.zeros(shape=[self.no], dtype=np.float32)

        self.ihWeights = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)
        self.hoWeights = np.zeros(shape=[self.nh, self.no], dtype=np.float32)

        self.hBiases = np.zeros(shape=[self.nh], dtype=np.float32)
        self.oBiases = np.zeros(shape=[self.no], dtype=np.float32)

        self.rnd = random.Random(seed)
        self.initializeWeights()

    # Store the Weights value into weights array.
    def setWeights(self, weights):
        if len(weights) != self.totalWeights(self.ni, self.nh, self.no):
            print("len(weights) error in setWeights()")

        index = 0

        for i in range(self.ni):
            for j in range(self.nh):
                self.ihWeights[i, j] = weights[index]
                index += 1

        for j in range(self.nh):
            self.hBiases[j] = weights[index]
            index += 1

        for j in range(self.nh):
            for k in range(self.no):
                self.hoWeights[j, k] = weights[index]
                index += 1

        for k in range(self.no):
            self.oBiases[k] = weights[index]
            index += 1

    def getWeights(self):
        tw = self.totalWeights(self.ni, self.nh, self.no)
        result = np.zeros(shape=[tw], dtype=np.float32)
        index = 0

        for i in range(self.ni):
            for j in range(self.nh):
                result[index] = self.ihWeights[i, j]
                index += 1

        for j in range(self.nh):
            result[index] = self.hBiases[j]
            index += 1

        for j in range(self.nh):
            for k in range(self.no):
                result[index] = self.hoWeights[j, k]
                index += 1

        for k in range(self.no):
            result[index] = self.oBiases[k]
            index += 1

        return result

    def initializeWeights(self):
        numberOfWeights = self.totalWeights(self.ni, self.nh, self.no)
        weights = np.zeros(shape=[numberOfWeights], dtype=np.float32)
        lo = -0.01
        hi = 0.01
        for index in range(len(weights)):
            weights[index] = (hi - lo) * self.rnd.random() + lo
        self.setWeights(weights)

    def computeOutputs(self, xValues):
        hSums = np.zeros(shape=[self.nh], dtype=np.float32)
        oSums = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(self.ni):
            self.iNodes[i] = xValues[i]

        for j in range(self.nh):
            for i in range(self.ni):
                hSums[j] += self.iNodes[i] * self.ihWeights[i, j]

        for j in range(self.nh):
            hSums[j] += self.hBiases[j]

        for j in range(self.nh):
            self.hNodes[j] = self.hypertan(hSums[j])

        for k in range(self.no):
            for j in range(self.nh):
                oSums[k] += self.hNodes[j] * self.hoWeights[j, k]

        for k in range(self.no):
            oSums[k] += self.oBiases[k]

        softOut = self.softmax(oSums)

        for k in range(self.no):
            self.oNodes[k] = softOut[k]

        result = np.zeros(shape=self.no, dtype=np.float32)
        for k in range(self.no):
            result[k] = self.oNodes[k]

        return result

    def train(self, trainData, maxEpochs, learnRate):
        hoGrads = np.zeros(shape=[self.nh, self.no], dtype=np.float32)
        # hidden layer -> output weights gradients

        obGrads = np.zeros(shape=[self.no], dtype=np.float32)
        # output node biases gradients

        ihGrads = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)
        # input layer -> hidden layer weights gradients

        hbGrads = np.zeros(shape=[self.nh], dtype=np.float32)
        # hidden biases gradients

        oSignals = np.zeros(shape=[self.no], dtype=np.float32)
        hSignals = np.zeros(shape=[self.nh], dtype=np.float32)

        epoch = 0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)
        numberOfTrainItems = len(trainData)
        indices = np.arange(numberOfTrainItems)  # [0, 1, 2, ....., n-1]

        while epoch < maxEpochs:
            self.rnd.shuffle(indices)
            for ii in range(numberOfTrainItems):
                index = indices[ii]

                for j in range(self.ni):
                    x_values[j] = trainData[index, j]  # get the input values
                for j in range(self.no):
                    t_values[j] = trainData[index, j + self.ni]  # get the target values
                self.computeOutputs(x_values)  # results stored in internal variable

                # 1. compute output node signals
                for k in range(self.no):
                    derivative = (1 - self.oNodes[k]) * self.oNodes[k]  # softmax函數
                    oSignals[k] = derivative * (self.oNodes[k] - t_values[k])  # E = (t - o)^2 do E' = (o - t)

                # 2. compute hidden layer -> output weight gradients using output signals
                for j in range(self.nh):
                    for k in range(self.no):
                        hoGrads[j, k] = oSignals[k] * self.hNodes[j]

                # 3. compute output layer node bias gradients using output signals
                for k in range(self.no):
                    obGrads[k] = oSignals[k] * 1.0

                # 4. compute hidden layer node signals
                for j in range(self.nh):
                    sumNum = 0.0
                    for k in range(self.no):
                        sumNum += oSignals[k] * self.hoWeights[j, k]
                    derivative = (1 - self.hNodes[j]) * (1 + self.hNodes[j])  # tanh activation
                    hSignals[j] = derivative * sumNum

                # 5. compute input layer to hidden layer weight gradients using hidden layer signals
                for i in range(self.ni):
                    for j in range(self.nh):
                        ihGrads[i, j] = hSignals[j] * self.iNodes[i]

                # 6. compute hidden layer node bias gradients using hidden layer signals
                for j in range(self.nh):
                    hbGrads[j] = hSignals[j] * 1.0

                # Update weights and biases using the gradients

                # 1. update input layer to hidden layer weights
                for i in range(self.ni):
                    for j in range(self.nh):
                        delta = -1.0 * learnRate * ihGrads[i, j]
                        self.ihWeights[i, j] += delta

                # 2. update hidden layer node biases
                for j in range(self.nh):
                    delta = -1.0 * learnRate * hbGrads[j]
                    self.hBiases[j] += delta

                # 3. update hidden layer to output layer weights
                for j in range(self.nh):
                    for k in range(self.no):
                        delta = -1.0 * learnRate * hoGrads[j, k]
                        self.hoWeights[j, k] += delta

                # 4. update output layer node biases
                for k in range(self.no):
                    delta = -1.0 * learnRate * obGrads[k]
                    self.oBiases[k] += delta

            epoch += 1

            if epoch % 10 == 0:
                mse = self.meanSquaredError(trainData)
                print("epoch = " + str(epoch) + " ms error = %0.4f " % mse)

                # end while

        result = self.getWeights()
        return result

    # end training


    def accuracy(self, tdata):  # train or test data matrix
        num_correct = 0
        num_wrong = 0

        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(len(tdata)):  # run each data items
            for j in range(self.ni):  # peel off imput values from current data row
                x_values[j] = tdata[i, j]  #
            for j in range(self.no):  # peel off target values from current data row
                t_values[j] = tdata[i, j + self.ni]

            y_values = self.computeOutputs(x_values)  # compute output values
            max_index = np.argmax(y_values)  # index of the largest value

            if abs(t_values[max_index] - 1.0) < 1.0e-5:
                num_correct += 1
            else:
                num_wrong += 1

        return (num_correct * 1.0) / (num_correct + num_wrong)

    def meanSquaredError(self, tdata):  # on train or test data matrix
        sumSquaredError = 0.0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)

        # same as for loop in "accuracy function"
        for ii in range(len(tdata)):
            for jj in range(self.ni):
                x_values[jj] = tdata[ii, jj]
            for jj in range(self.no):
                t_values[jj] = tdata[ii, jj + self.ni]

            y_values = self.computeOutputs(x_values)

            for j in range(self.no):
                err = t_values[j] - y_values[j]
                sumSquaredError += err * err  # (t-o)^2

        return sumSquaredError / len(tdata)

    @staticmethod
    def hypertan(x):
        if x < -20.0:
            return -1.0
        elif x > 20.0:
            return 1.0
        else:
            return math.tanh(x)

    @staticmethod
    def softmax(oSums):
        result = np.zeros(shape=[len(oSums)], dtype=np.float32)
        m = max(oSums)
        divisor = 0.0
        for k in range(len(oSums)):
            divisor += math.exp(oSums[k] - m)
        for k in range(len(result)):
            result[k] = math.exp(oSums[k] - m) / divisor
        return result

    @staticmethod
    def totalWeights(nInput, nHidden, nOutput):
        totalweight = (nInput * nHidden) + (nHidden * nOutput) + nHidden + nOutput
        return totalweight

        # end class Neural Network


def main():
    print("\nBegin BackPropagation demo")

    numInput = 4
    numHidden = 5
    numOutput = 3

    neuralNetwork = NeuralNetwork(numInput, numHidden, numOutput, seed=3)

    trainDataPath = "irisTrainData.txt"
    trainDataMatrix = loadFile(trainDataPath)

    showMatrixPartial(trainDataMatrix, 4, 1, True)
    testDataPath = "irisTestData.txt"
    testDataMatrix = loadFile(testDataPath)

    maxEpochs = 50
    learnRate = 0.05
    neuralNetwork.train(trainData=trainDataMatrix, maxEpochs=maxEpochs, learnRate=learnRate)

    trainAccuracy = neuralNetwork.accuracy(trainDataMatrix)
    testAccuracy = neuralNetwork.accuracy(testDataMatrix)

    print("\nAccuracy on 120 training data = %0.4f " % trainAccuracy)
    print("Accuracy on 30 testing data  = %0.4f " % testAccuracy)


if __name__ == "__main__":
    main()
