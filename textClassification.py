#LOAD DATASET
def loadDataset():
    
    import json
    
    with open('News Classification DataSet.json', 'r') as handle:
        data = [json.loads(line) for line in handle]
        
    dataset = [[data[i]['content'], data[i]['annotation']['label'][0]] for i in range(len(data))]
    
    return dataset


#SPLIT DATASET INTO TRAINING AND TESTING
def splitDataIntoTrainAndTest(dataset):
    """
        Split The Dataset into Training and Test Set
        Training Set = 70% of Dataset
        TestSet = 30% of Dataset
    """
    import random

    size = len(dataset)  #get size of dataset
    trainSize = (size*70)//100  #Training Set consists of 70% of total entries
    testSize = size - trainSize
    
    trainSet = list(range(size))     #Stores Indices of TrainingSet
    testSet = []                        #Stores Indices of TestSet
    for i in range(testSize):
        randIndex = int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    
    trainMat = []       #Stores Training Dataset
    testMat = []        #Stores Test Dataset
    
    for i in range(trainSize):
        trainMat.append(dataset[trainSet[i]])
        
    for i in range(testSize):
        testMat.append(dataset[testSet[i]])
        
    return trainMat,testMat


#PREPROCESS DATASET
def preprocess(dataset):
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.stem import PorterStemmer
    from nltk.tokenize import RegexpTokenizer
    from gensim.corpora import Dictionary

    stopwordsList1 = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 
    'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 
    'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 
    'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 
    'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 
    'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 
    'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 
    'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 
    'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 
    'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 
    'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 
    'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 
    'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 
    'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 
    'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 
    'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 
    'having', 'he', 'her', 'here', 'herself', 'high', 'high', 'high', 'higher', 'highest', 
    'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 
    'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 
    'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 
    'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer',
    'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 
    'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 
    'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer',
    'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 
    'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on',
    'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 
    'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 
    'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point',
    'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 
    'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 
    'really', 'right', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 
    'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 
    'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 
    'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 
    'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 
    'still', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 
    'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 
    'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 
    'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 
    'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 
    'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 
    'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 
    'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 
    'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 
    'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 
    'yours', 'z']
    stopwordsList2 = list(stopwords.words('english'))
    lemma = WordNetLemmatizer()    
    stemma = PorterStemmer()

    corpus = []
    emptydocs = set()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]\w*')
    for i in range(len(dataset)):
        doc = dataset[i][0]
        tokenizeddoc = tokenizer.tokenize(doc)
        processeddoc = []
        for j in range(len(tokenizeddoc)):
            word = tokenizeddoc[j]
            if len(word) <= 2:
                continue
            word = word.lower()
            word = stemma.stem(lemma.lemmatize(word))
            if word in stopwordsList1 or word in stopwordsList2:
                continue
            processeddoc.append(word)
        if len(processeddoc)==0:
            emptydocs.add(i)
            continue
        corpus.append(processeddoc)
    
    for i in reversed(sorted(list(emptydocs))):
        del dataset[i]
    
    dictionary = Dictionary(corpus) #Bag of words. Create a dictionary containing the word and words unique ID.
    dictionary.filter_extremes(no_below=2)
    word2id = dictionary.token2id
    #Manual filtering.
    if 'apf' in word2id:
        del word2id['afp'] #Remove newspaper's name
    if 'ap' in word2id:
        del word2id['ap'] #Remove newspaper's name
    if 'cp' in word2id:
        del word2id['cp'] #Remove newspaper's name
    if 'newsday' in word2id:
        del word2id['newsday'] #Remove newspaper's name
    if 'reuter' in word2id:
        del word2id['reuter'] #Remove newspaper's name
    if 'usatoday' in word2id:
        del word2id['usatoday'] #Remove newspaper's name]
    
    filteredcorpus = []
    emptydocs = set()
    for i in range(len(corpus)):
        tokenizeddoc = corpus[i]
        processeddoc = []
        for j in range(len(tokenizeddoc)):
            word = tokenizeddoc[j]
            if word not in word2id:
                continue
            processeddoc.append(word)
        if len(processeddoc)==0:
            emptydocs.add(i)
            continue
        filteredcorpus.append(processeddoc)
    
    for i in reversed(sorted(list(emptydocs))):
        del dataset[i]
    
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i][1])
    
    corpus = filteredcorpus
    return  corpus, labels, word2id


#CREATE MODEL
def word2Vec(corpus,numfeat):
    
    from gensim.models import Word2Vec
    
    model = Word2Vec(sentences=corpus, size=numfeat, window=5, min_count=1, workers=2, sg=1)
    vocabulary = list(model.wv.vocab) # summarize vocabulary
    
    return model, vocabulary


#PLOT MODEL
def plotModel(model):
    from sklearn.decomposition import PCA
    from matplotlib import pyplot
    
    X = model[model.wv.vocab]
    pca = PCA(n_components=2) # fit a 2d PCA model to the vectors
    result = pca.fit_transform(X)
    pyplot.scatter(result[:, 0], result[:, 1]) # create a scatter plot of the projection
    for i, word in enumerate(vocabulary):
    	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


'''
#APPLY TFIDF
def tfidf(corpus,dictionary):
    
    from gensim.models import TfidfModel
    
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus] #For each document a list of tuples is created reporting the words(ID) in filtered dictionary and frequency of those words.
    tfidf = gensim.models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]
    
    return tfidf_corpus,tfidf
'''
 

#CREATE TRAINING DOCUMENT VECTORS
def getTrainingDocVector(model,corpus,numfeat):
    
    featvec = []
    for i in range(len(corpus)):
        doc = corpus[i]
        docvec = [0 for j in range(numfeat)]
        for j in range(len(doc)):
            word = doc[j]
            wordvec = model[word]
            docvec = [docvec[k] + wordvec[k] for k in range(numfeat)]
        docvec = [docvec[j]/len(doc) for j in range(numfeat)]
        featvec.append(docvec)
        
    return featvec
    

#CREATE TESTING DOCUMENT VECTORS
def getTestingDocVector(model,corpus,numfeat):
    
    featvec = []
    for i in range(len(corpus)):
        doc = corpus[i]
        docvec = [0 for j in range(numfeat)]
        for j in range(len(doc)):
            word = doc[j]
            if word not in word2id:
                continue;
            wordvec = model[word]
            docvec = [docvec[k] + wordvec[k] for k in range(numfeat)]
        docvec = [docvec[j]/len(doc) for j in range(numfeat)]
        featvec.append(docvec)
        
    return featvec


#KNN ALGORITHM
#@jit(nopython=True, parallel=True)
def classifyKNN(inX,trainMat,labels,k):
    
    import operator
    
    trainMatSize = trainMat.shape[0]
    diffMat = tile(inX, (trainMatSize,1)) - trainMat
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount={}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        #print(voteLabel)
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


if __name__=="__main__":
    
    from numpy import *
    
    dataset = loadDataset()
    trainDataset, testDataset = splitDataIntoTrainAndTest(dataset)
    trainCorpus, trainLabels, word2id = preprocess(trainDataset)
    numfeat = 100
    model, vocabulary = word2Vec(trainCorpus,numfeat)
    trainMat = getTrainingDocVector(model,trainCorpus,numfeat)
    testCorpus, testLabels, waste = preprocess(testDataset)
    testMat = getTestingDocVector(model,testCorpus,numfeat)
    
    results = []
    for i in range(len(testMat)):
        vec = testMat[i]
        result = classifyKNN(vec,asarray(trainMat),trainLabels,numfeat)
        print(i, testLabels[i], result)
        results.append(result)
    
    error = 0;
    for i in range(len(results)):
        if results[i] != testLabels[i]:
            error += 1
    print(error)
    accuracy = (len(trainMat)-error)/len(trainMat)
    print(accuracy)



"""
  #####  NAIVE BAYES   #####

"""

def createVocabList(dataset):
    vocabset = set([])
    for document in dataset:
        vocabset = vocabset | set(document)
    return list(vocabset)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0.0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("The word %s is not in my vocabulory" %word)
    return returnVec
    
def training(trainMatrix,trainLabels):
    
    numTrainDocs = len(trainMatrix)
    classProb = {}
    for i in range(len(trainLabels)):
        classProb[trainLabels[i]] = classProb.get(trainLabels[i],0) + 1
    
    for label,freq in classProb.items():
        classProb[label] /= numTrainDocs
            
    numWords = len(trainMatrix[0])
    
    pNum = {}         # to store Nk + 1 for each word of each class
    pDenom = {}       # to stroe N + |vocab| for each class
    
    for label,temp in classProb.items():
        
        tempNum = ones(numWords)
        tempDenom = numWords
        
        pNum[label] = tempNum
        pDenom[label] = tempDenom
    
    for i in range(numTrainDocs):
        curClass = trainLabels[i]
        pNum[curClass] += trainMatrix[i]
        pDenom[curClass] += sum(trainMatrix[i])
    
    classConditionalProb = {} 
    
    for label,count in pNum.items():
        classConditionalProb[label] = [log(pNum[label][i]/pDenom[label]) for i in range(numWords)]
    
    return classProb,classConditionalProb


def classify(inpVector,classProb,classCondProb):
    
    prevVal = -10000000000
    ans = "temp"
    
    for label,value in classProb.items():
        
        val = 0.0
        for i in range(len(inpVector)):
            val += inpVector[i]*classCondProb[label][i]
        val += log(classProb[label])
        if val > prevVal:
            prevVal = val
            ans = label
    
    return ans



dataset = loadDataset()
trainDataset, testDataset = splitDataIntoTrainAndTest(dataset)
trainCorpus, trainLabels, word2id = preprocess(trainDataset)
testCorpus, testLabels, waste = preprocess(testDataset)

trainMat = []
vocabulary = createVocabList(trainCorpus)
for i in range(len(trainCorpus)):
    trainMat.append(setOfWords2Vec(vocabulary,trainCorpus[i]))
classProb,classCondProb = training(trainMat,trainLabels)

errorCount = 0.0
results = []
for i in range(len(testCorpus)):
    testVec = setOfWords2Vec(vocabulary,testCorpus[i])
    result = classify(testVec,classProb,classCondProb)
    results.append(result)
    if result != testLabels[i]:
        errorCount += 1.0

print(errorCount/len(testCorpus))


# ANN

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import np_utils


dataset = loadDataset()
trainDataset, testDataset = splitDataIntoTrainAndTest(dataset)
trainCorpus, trainLabels, word2id = preprocess(trainDataset)
numfeat = 20
model, vocabulary = word2Vec(trainCorpus,numfeat)
trainMat = getTrainingDocVector(model,trainCorpus,numfeat)
testCorpus, testLabels, waste = preprocess(testDataset)
testMat = getTestingDocVector(model,testCorpus,numfeat)

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()

y_train =  encoder.fit_transform(trainLabels)
y_test =  encoder.fit_transform(testLabels)

from numpy import *
classifier = Sequential()
#Adding input layer and 1st hidden layer
classifier.add(Dense(output_dim = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 20))
classifier.add(Dropout(p = 0.15))
#Adding second hidden layer
#classifier.add(Dense(output_dim = 20, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(p = 0.15))
#Adding Output Layer
classifier.add(Dense(output_dim = 4, kernel_initializer = 'uniform', activation = 'softmax'))

#Comiling the ANN
classifier.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

trainMat = array(trainMat)
testMat = array(testMat)

classifier.fit(trainMat,y_train,nb_epoch = 100,batch_size = 20)
score = classifier.evaluate(testMat,y_test)

'''
#LOAD DATASET
import pandas as pd

dataextract = pd.read_csv('uci-news-aggregator.csv')
col1 = dataextract['TITLE']
col2 = dataextract['CATEGORY'] #b : business, t : science and technology, e : entertainment, m : health and medicine
dataset = []
for i in range(len(dataextract)):
    news = []
    news.append(col1[i])
    news.append(col2[i])
    dataset.append(news)
del dataextract
del col1
del col2
del i
del news


#NORMALIZE
def autonorm(dataset):
    dataset = array(dataset)
    minVals = dataset.min(axis = 0)
    maxVals = dataset.max(axis = 0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataset))
    n = dataset.shape[0]
    normDataset = dataset - tile(minVals,(n,1))
    normDataset = normDataset / tile(maxVals,(n,1))
    return normDataset,ranges,minVals


'''


#ADABOOST 


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    
    returnArr = ones((shape(dataMatrix)[0],1))
    
    if threshIneq == 'lt':
        returnArr[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        returnArr[dataMatrix[:,dimen] >= threshVal] = -1.0

    return returnArr


def buildStump(dataArr,classLabels,D):
    
    dataMatrix = mat(dataArr)
    labelMatrix = mat(classLabels).T
    m,n = shape(dataArr)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = rangeMin + float(j)*stepSize
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMatrix] = 0
                weightedError = D.T * errArr
                #print("dimension : %s Thresh : %f Error : %f" %(i,threshVal,weightedError))            
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                
    return bestStump,minError,bestClassEst
    

def adaboostTrain(dataArr,classLabels,numIt = 10):
    
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    
    aggClassEst = mat(zeros((m,1)))
    
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        alpha = float(0.5*log((1.0-error)/max(error,1e-6)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        aggErrors = multiply(mat(classLabels).T != sign(aggClassEst),mat(ones((m,1))))
        errorRate = aggErrors.sum()/m
        #print(errorRate)
        if errorRate == 0.0:
            break
        
    return weakClassArr


def adaboostTest(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print(aggClassEst)
        
    return aggClassEst





def adaboost():
    
    dataset = loadDataset()
    trainDataset, testDataset = splitDataIntoTrainAndTest(dataset)
    trainCorpus, trainLabels, word2id = preprocess(trainDataset)
    testCorpus, testLabels, waste = preprocess(testDataset)
    numfeat = 20
    model, vocabulary = word2Vec(trainCorpus,numfeat)
    trainMat = getTrainingDocVector(model,trainCorpus,numfeat)
    testMat = getTestingDocVector(model,testCorpus,numfeat)
    
    binaryTrainingLabels = []
    for i in range(len(trainLabels)):
        if trainLabels[i]=='SciTech':
            binaryTrainingLabels.append(1)
        else:
            binaryTrainingLabels.append(-1)
            
    dict1 = adaboostTrain(trainMat,binaryTrainingLabels)
    val1 = adaboostTest(testMat,dict1)
    
    binaryTrainingLabels = []
    for i in range(len(trainLabels)):
        if trainLabels[i]=='Sports':
            binaryTrainingLabels.append(1)
        else:
            binaryTrainingLabels.append(-1)
    dict2 = adaboostTrain(trainMat,binaryTrainingLabels)
    val2 = adaboostTest(testMat,dict2)
    
    binaryTrainingLabels = []
    for i in range(len(trainLabels)):
        if trainLabels[i]=='World':
            binaryTrainingLabels.append(1)
        else:
            binaryTrainingLabels.append(-1)
    dict3 = adaboostTrain(trainMat,binaryTrainingLabels)
    val3 = adaboostTest(testMat,dict3)
    
    binaryTrainingLabels = []
    for i in range(len(trainLabels)):
        if trainLabels[i]=='Business':
            binaryTrainingLabels.append(1)
        else:
            binaryTrainingLabels.append(-1)
    dict4 = adaboostTrain(trainMat,binaryTrainingLabels)
    val4 = adaboostTest(testMat,dict4)
    
    























