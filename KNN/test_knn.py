import KNN as knn

if __name__ == '__main__':
    hoRatio = 0.01 # hold out 10%
    datingDataMat, datingLabels = knn.file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = knn.autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = knn.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with:"+str(classifierResult), "the real answer is:"+str(datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: "+str((errorCount / float(numTestVecs))))


