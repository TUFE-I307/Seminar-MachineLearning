# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression

""""
函数说明:使用Sklearn构建Logistic回归分类器

Parameters:
	无
Returns:
	无
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-09-05
"""
def colicSklearn():
	frTrain = open('horseColicTraining.txt')										#打开训练集
	frTest = open('horseColicTest.txt')												#打开测试集
	trainingSet = []; trainingLabels = []
	testSet = []; testLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	for line in frTest.readlines():
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		testSet.append(lineArr)
		testLabels.append(float(currLine[-1]))
	classifier = LogisticRegression(solver = 'sag',max_iter = 5000).fit(trainingSet, trainingLabels)
	test_accurcy = classifier.score(testSet, testLabels) * 100
	print('正确率:%f%%' % test_accurcy)

if __name__ == '__main__':
	colicSklearn()
