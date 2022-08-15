import random
import os
import copy
import math
import numpy as np
import sys


class Model:

    def __init__(self, K, V, iterNum, alpha, beta, dataset, ParametersStr, sampleNo,
                 wordsInTopicNum):
        self.K = K
        self.V = V
        self.iterNum = iterNum
        self.dataset = dataset
        self.ParametersStr = ParametersStr
        self.sampleNo = sampleNo
        self.wordsInTopicNum = copy.deepcopy(wordsInTopicNum)

        self.alpha = alpha
        self.beta = beta
        

        self.beta0 = float(V) * float(beta)

        self.smallDouble = 1e-150
        self.largeDouble = 1e150


        self.large = 1e150
        self.small = 1e-150

    def run_ICMM_withGMM(self, documentSet, outputPath, wordList):
        # The whole number of documents
        self.D_All = documentSet.D  # document的总数
        # Cluster assignments of each document               (documentID -> clusterID)
        # 将每一个docu分配到某个聚类中，初始的聚类id都为-1
        self.z = [-1] * self.D_All
        # The number of documents in cluster z               (clusterID -> number of documents)
        # 聚类中docu的数量
        self.m_z = [0] * self.K
        # The number of words in cluster z                   (clusterID -> number of words)
        # 聚类中word的数量
        self.n_z = [0] * self.K
        # The number of occurrences of word v in cluster z   (n_zv[clusterID][wordID] = number)
        # word v在聚类z中出现的次数
        self.n_zv = [[0] * self.V for _ in range(self.K)]
        # different from K, K is clusterID but K_current is clusterNum
        # 聚类的数量
        self.K_current = copy.deepcopy(self.K)
        # word list in initialization
        # 初始化word列表
        self.word_current = []

        self.intialize(documentSet)
        self.gibbsSampling(documentSet)
        print("\tGibbs sampling successful! Start to saving results.")
        self.output(documentSet, outputPath, wordList)
        print("\tSaving successful!")

    # Get beta0 for current V
    def getBeta0(self):
        return (float(len(list(set(self.word_current)))) * float(self.beta))

    def intialize(self, documentSet):
        self.alpha0 = self.alpha * self.D_All
        print("\t" + str(self.D_All) + " documents will be analyze. alpha is" + " %.2f." % self.alpha +
              " beta is" + " %f." % self.beta +
              "\n\tInitialization.")

        for d in range(0, self.D_All):
            document = documentSet.documents[d]
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                self.word_current.append(wordNo)
            self.beta0 = self.getBeta0()
            if self.beta0 <= 0:
                print("Wrong V!")
                exit(-1)

            '''1 全部按照最大概率'''
            cluster = self.sampleCluster(-1, document, "Max")
            # print("cluster={}, current_k={}, docu={}".format(cluster, self.K_current, d))

            self.z[d] = cluster
            if cluster == len(self.m_z):
                '''
                如果聚类的id等于已有该列表长度
                说明有新的聚类出现，因此，以下所有的长度都要增加1
                '''
                self.m_z.append(0)
                self.n_zv.append([0] * self.V)
                self.n_z.append(0)
                
            self.m_z[cluster] += 1
            
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                # 更新每个聚类中，每个词的数量
                self.n_zv[cluster][wordNo] += wordFre
                # 更新每个聚类中word的数量
                self.n_z[cluster] += wordFre

    def gibbsSampling(self, documentSet):
        for i in range(self.iterNum):
            print("\titer is ", i + 1, end="\t")
            print("beta is" + " %f." % self.beta, end='\t')
            print("Kcurrent is" + " %f." % self.K_current, end='\n')
            for d in range(0, self.D_All):
                document = documentSet.documents[d]
                cluster = self.z[d]
                self.m_z[cluster] -= 1
                
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                self.checkEmpty(cluster)

                if i != self.iterNum - 1:
                    cluster = self.sampleCluster(i, document, "Max")
                else:
                    cluster = self.sampleCluster(i, document, "Max")

                

                self.z[d] = cluster
                if cluster == len(self.m_z):
                    '''
                    如果聚类的id等于已有该列表长度
                    说明有新的聚类出现，因此，以下所有的长度都要增加1
                    '''
                    self.m_z.append(0)
                    self.n_zv.append([0] * self.V)
                    self.n_z.append(0)
                    
                self.m_z[cluster] += 1
                
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre

    def sumNormalization(self, x):
        """Normalize the prob.
        正则化
        """
        x = np.array(x)
        norm_x = x / np.sum(x)
        return norm_x

    '''
    MODE
    "Max"  Choose cluster with max probability.
    "Random"  Random choose a cluster accroding to the probability.
    '''

    def sampleCluster(self, _iter, document, MODE):
        prob = [float(0.0)] * (self.K + 1)
        overflowCount = [float(0.0)] * (self.K + 1)
        overflowCount_2 = [float(0.0)] * (self.K + 1)
        prob_2 = [float(0.0)] * (self.K + 1)
        e_index = [float(0.0)] * (self.K + 1)

        for k in range(self.K):
            if self.m_z[k] == 0:
                prob[k] = 0
                prob_2[k] = 0
                continue
            # DPMM
            
            # valueOfRule1 = self.m_z[k] # / (self.D_All - 1 + self.alpha0)
            valueOfRule1 = self.m_z[k] / (self.D_All - 1 + self.alpha0)
            valueOfRule2 = 1.0
            i = 0
            for _, w in enumerate(range(document.wordNum)):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if valueOfRule2 < self.smallDouble:
                        overflowCount[k] -= 1
                        valueOfRule2 *= self.largeDouble
                    '''
                    valueOfRelu2可能有问题
                    不是哦, j和i都是从0开始的， 没问题这里
                    valueOfRule2 *= (self.n_zv[k][wordNo] + self.beta + j - 1) / (self.n_z[k] + self.beta0 + i - 1)
                    '''
                    valueOfRule2 *= (self.n_zv[k][wordNo] + self.beta + j) / (self.n_z[k] + self.beta0 + i)
                    i += 1
            prob[k] = valueOfRule1 * valueOfRule2

            

        '''==============================================================================='''

        '''the prob of new cluster'''
        # DPMM
        
        # valueOfRule1 = self.alpha # / (self.D_All - 1 + self.alpha0)
        valueOfRule1 = self.alpha0 / (self.D_All - 1 + self.alpha0)
        valueOfRule2 = 1.0
        i = 0
        for _, w in enumerate(range(document.wordNum)):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                if valueOfRule2 < self.smallDouble:
                    overflowCount[self.K] -= 1
                    valueOfRule2 *= self.largeDouble
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] = valueOfRule1 * valueOfRule2
        


        max_overflow = -sys.maxsize
        
        for k in range(self.K + 1):
            if overflowCount[k] > max_overflow and prob[k] > 0.0:
                max_overflow = overflowCount[k]



        for k in range(self.K + 1):
            '''
            防止概率过小，
            前面过小的概率乘以了多次largeDouble, 这里要把它变回来。
            '''
            if prob[k] > 0.0:
                prob[k] = prob[k] * math.pow(self.largeDouble, overflowCount[k] - max_overflow)

        
        prob = self.sumNormalization(prob)  # DPMM
        

        

        if MODE == "Random":
            kChoosed = 0
            for k in range(1, self.K + 1):
                prob[k] += prob[k - 1]
            thred = random.random() * prob[self.K]
            while kChoosed < self.K + 1:
                if thred < prob[kChoosed]:
                    break
                kChoosed += 1
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
            return kChoosed

        elif MODE == "Max":
            kChoosed = 0
            bigPro = prob[0]
            for k in range(1, self.K + 1):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
            return kChoosed

    # update K_current
    def checkEmpty(self, cluster):
        '''
        更新聚类的数量，如果某个cluster中docu的数量等于0，那么聚类数量减1
        '''
        if self.m_z[cluster] == 0:
            self.K_current -= 1

    def output(self, documentSet, outputPath, wordList):
        '''
        输出一些重要信息，根据每个输出函数输出所需信息
        '''
        outputDir = outputPath + self.dataset + self.ParametersStr + "/"
        try:
            # create result/
            isExists = os.path.exists(outputPath)
            if not isExists:
                os.mkdir(outputPath)
                print("\tCreate directory:", outputPath)
            # create after result
            isExists = os.path.exists(outputDir)
            if not isExists:
                os.mkdir(outputDir)
                print("\tCreate directory:", outputDir)
        except:
            print("ERROR: Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet)
        self.estimatePosterior()
        try:
            self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        except:
            print("\tOutput Phi Words Wrong!")
        self.outputSizeOfEachCluster(outputDir, documentSet)

    def estimatePosterior(self):  # φ
        self.phi_zv = [[0] * self.V for _ in range(self.K)]  # k * v维数组
        for cluster in range(self.K):
            for v in range(self.V):
                self.phi_zv[cluster][v] = float(self.n_zv[cluster][v] + self.beta) / float(
                    self.n_z[cluster] + self.beta0)

    def getTop(self, array, rankList, Cnt):
        '''
        得到最好的Cnt个结果，放在ranklist中
        '''
        index = 0
        m = 0
        while m < Cnt and m < len(array):
            max = 0
            for no in range(len(array)):
                if (array[no] > max and no not in rankList):
                    index = no
                    max = array[no]
            rankList.append(index)
            m += 1

    def outputPhiWordsInTopics(self, outputDir, wordList, Cnt):
        '''
        输出每个聚类中，出现次数最多的Cnt个words
        '''
        outputfiledir = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "PhiWordsInTopics.txt"
        writer = open(outputfiledir, 'w')
        for k in range(self.K):
            rankList = []
            if self.m_z[k] == 0:
                continue
            topicline = "Topic " + str(k) + ":\n"
            writer.write(topicline)
            self.getTop(self.phi_zv[k], rankList, Cnt)
            for i in range(rankList.__len__()):
                tmp = "\t" + wordList[rankList[i]] + "\t" + str(self.phi_zv[k][rankList[i]])
                writer.write(tmp + "\n")
        writer.close()

    def outputSizeOfEachCluster(self, outputDir, documentSet):
        '''
        输出每一个cluster的大小
        '''
        outputfile = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "SizeOfEachCluster.txt"
        writer = open(outputfile, 'w')
        topicCountIntList = []
        for cluster in range(self.K):
            if self.m_z[cluster] != 0:
                topicCountIntList.append([cluster, self.m_z[cluster]])
        line = ""
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n\n")
        line = ""
        topicCountIntList.sort(key=lambda tc: tc[1], reverse=True)
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n")
        writer.close()

    def outputClusteringResult(self, outputDir, documentSet):
        '''输出聚类结果
            document属于某一个cluster
        '''
        outputPath = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "ClusteringResult" + ".txt"
        writer = open(outputPath, 'w')
        for d in range(0, self.D_All):
            documentID = documentSet.documents[d].documentID
            cluster = self.z[d]
            writer.write(str(documentID) + " " + str(cluster) + "\n")
        writer.close()
