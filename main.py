"""
DPM
"""
 
from DPM import DPM
import time
import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--Lambda', type=float, default=0.0005,
                        help='value of param')

args = parser.parse_args()

K = 0                  #初始化簇数
sampleNum = 3          #重复试验次数
iterNum = 15          #单次试验模型迭代次数
wordsInTopicNum = 20   #topN个簇代表词

alpha = 0.01         #CMM先验参数1
beta = args.Lambda          #CMM先验参数2



# dataset = "Tweet-SIMCSE"
# dataset = "GoogleNews-SIMCSE"
# dataset = "Tweet-SIMCSE"
# dataset = "R52-SIMCSE"
# dataset = "20ng-SIMCSE"
# dataset = "Biomedical-SIMCSE"
# dataset = "StackOverflow-SIMCSE"
# dataset = "News-S-SIMCSE"
# dataset = "News-TS-SIMCSE"
# dataset = "AgNews-SIMCSE"
# dataset = "Search-Snippets-SIMCSE"
dataset = "20ng-SIMCSE"

dataDir = "../data/"
outputPath = "result/20ng_beta_S3_iter15/"


def runDPM(K, alpha, beta, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir):
    dpm = DPM(K, alpha, beta, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir)
    '''
    返回DPM对象
    ICMM_with有两个函数
    getDocuments(self)
    runDPM(self, sampleNo, outputPath)
    '''
    
    dpm.getDocuments()
    for sampleNo in range(1, sampleNum + 1):
        print("SampleNo:" + str(sampleNo))
        dpm.runDPM(sampleNo, outputPath)



if __name__ == '__main__':
    outf = open("time_DPM", "a")
    time1 = time.time()
    runDPM(K, alpha, beta, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir)
    time2 = time.time()
    outf.write(str(dataset) + "K" + str(K) + "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3)) +
               "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) +
               "\ttime:" + str(time2 - time1) + "\n")





