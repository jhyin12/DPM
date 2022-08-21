from DocumentSet import DocumentSet
from Model import Model


class DPM:

    def __init__(self, K, alpha, beta, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        
        self.iterNum = iterNum
        self.sampleNum = sampleNum
        self.dataset = dataset
        self.wordsInTopicNum = wordsInTopicNum
        self.dataDir = dataDir

        self.wordList = []
        self.wordToIdMap = {}

    def getDocuments(self):
        self.documentSet = DocumentSet(self.dataDir + self.dataset, self.wordToIdMap, self.wordList)
        '''
        返回DocumentSet对象
        DocumentSet有一个初始化方法
        '''
        self.V = self.wordToIdMap.__len__() 
        '''
        v是词的总个数
        '''

    def runDPM(self, sampleNo, outputPath):
        ParametersStr = "K" + str(self.K) + "alpha" + str(round(self.alpha, 3)) + "beta" + str(round(self.beta, 3)) + \
                        "iterNum" + str(self.iterNum) + "SampleNum" + str(self.sampleNum)
        model = Model(self.K, self.V, self.iterNum, self.alpha, self.beta,
                      self.dataset, ParametersStr, sampleNo, self.wordsInTopicNum)
        model.run_DPM(self.documentSet, outputPath, self.wordList)
