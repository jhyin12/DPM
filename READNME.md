## experiment for ICMM_withGMM

* environment requirements : `numpy`, `sklearn`


#### train model
```
$ python main.py
```

`K` : 初始化簇数

`sampleNum` : 重复试验次数

`iterNum` : 单次试验模型迭代次数

`wordsInTopicNum` : topN个簇代表词

`alpha` : CMM先验参数1

`beta` : CMM先验参数2



`dataset` : 数据集

训练完成，将在result生成结果文件。

example：
```
result
└───Tweet-SIFK0alpha0.1beta0.1iterNum10SampleNum1
│   │   Tweet-SIFSampleNo1ClusteringResult.txt
│   │   Tweet-SIFSampleNo1PhiWordsInTopics.txt
│   │   Tweet-SIFSampleNo1SizeOfEachCluster.txt
```


#### 模型调优

```
参数范围：
K = 0
sampleNum = 5
iterNum = 10
wordsInTopicNum = 20
alpha = [0.01, 0.5], skip = 0.01
beta = [0.01, 0.5], skip = 0.01

dataset = [GoogleNews, Tweet, StackOverflow, R52, Biomedical, 20ng]

调参优先级：
beta > alpha
调参过程中可根据NMI得分（尽可能搜找最高点）、以及得分homogeneity和completeness的趋势判定是否继续下调。
尝试上述所有可能的组合，不是仅更改一项，固定其它的那种，是n*n*n...n这种。
```


#### evaluate model
```
$ python Evaluation.py
```
`K`, `sampleNum`, `iterNum`, `alpha`, `beta`, `dataset`, 所有参数变化跟随main.py

评估完成，将在result中对应的结果文件夹内生成评价得分。

example：
```
result
└───Tweet-SIFK0alpha0.1beta0.1iterNum10SampleNum1
│   │   ICMM_withGMMDatasetTweet-SIFK0alpha0.1beta0.1iterNum10SampleNum1NoiseKThreshold0.txt
```


#### show results
```
# ICMM_withGMM Tweet-SIF K0 iterNum10 SampleNum1 alpha0.1 beta0.1 

ACCMean:				[0.8288834951456311]
purityMean:				[0.9308252427184466]
ARIMean:				[0.8179238505615284]
AMIMean:				[0.8980564190057365]
NMIMean:				[0.9192860768133686]
homogeneityMean:		[0.9408733406890566]
completenessMean:		[0.8986671849612801]
VMean:					[0.9192860768133687]

ACCVariance:			[0.0]
purityVariance:			[0.0]
ARIVariance:			[0.0]
AMIVariance:			[0.0]
NMIVariance:			[0.0]
homogeneityVariance:	[0.0]
completenessVariance:	[0.0]
VVariance:				[0.0]

KRealNumMean:			[89.0]
KPredNumMean:			[115.0]

KRealNumVariance:		[0.0]
KPredNumVariance:		[0.0]

--------------------------------------------------

```


# Citation
If you find this project useful, please consider citing:

```bibtex
@INPROCEEDINGS{7498276,  
    author={Yin, Jianhua and Wang, Jianyong},  
    booktitle={2016 IEEE 32nd International Conference on Data Engineering (ICDE)},   
    title={A model-based approach for text clustering with outlier detection},   
    year={2016},  
    volume={},  
    number={},  
    pages={625-636},  
    doi={10.1109/ICDE.2016.7498276}}
```

