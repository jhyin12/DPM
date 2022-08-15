"""
ICMM_withGMM
"""
 
from ICMM_withGMM import ICMM_withGMM
import time

K = 0                  #初始化簇数
sampleNum = 3          #重复试验次数
iterNum = 15          #单次试验模型迭代次数
wordsInTopicNum = 20   #topN个簇代表词

alpha = 0.01           #CMM先验参数1
beta = 0.1           #CMM先验参数2



# dataset = "Tweet-SIMCSE"
# dataset = "GoogleNews-SIMCSE"
# dataset = "Tweet-SIMCSE"
# dataset = "R52-SIMCSE"
# dataset = "20ng-SIMCSE"
dataset = "Biomedical-SIMCSE"
# dataset = "StackOverflow-SIMCSE"

dataDir = "data/"
outputPath = "result/"


def runICMM_withGMM(K, alpha, beta, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir):
    icmm_withgmm = ICMM_withGMM(K, alpha, beta, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir)
    '''
    返回ICMM_withGMM对象
    ICMM_with有两个函数
    getDocuments(self)
    runICMM_withGMM(self, sampleNo, outputPath)
    '''
    
    icmm_withgmm.getDocuments()
    for sampleNo in range(1, sampleNum + 1):
        print("SampleNo:" + str(sampleNo))
        icmm_withgmm.runICMM_withGMM(sampleNo, outputPath)



if __name__ == '__main__':
    outf = open("time_ICMM_withGMM", "a")
    time1 = time.time()
    runICMM_withGMM(K, alpha, beta, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir)
    time2 = time.time()
    outf.write(str(dataset) + "K" + str(K) + "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3)) +
               "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) +
               "\ttime:" + str(time2 - time1) + "\n")





