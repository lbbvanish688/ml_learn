from numpy import *
import operator

def file2matrix(filename):
    fr=open(filename)
    numberOfLines=len(fr.readlines())
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    fr=open(filename)
    index=0
    for line in fr.readlines():
        line=line.strip()
        listFormLine=line.split('\t')
        returnMat [index,:]=listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=normDataSet-tile(minVals,(m,1))
    normDataSet= normDataSet/tile(ranges, (m,1))
    return normDataSet,ranges,minVals
    #分类函数
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


