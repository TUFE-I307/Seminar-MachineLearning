一，	鸢尾花实验
1，	变量说明
dataset 数据集
rate 训练集占比
train 训练集
test 测试集
2，	函数作用
randsplit(dataset,rate) 切分训练集和测试集
gauss_clsssify(train,test) 构建贝叶斯分类器

二，	侮辱类文本分类
1，变量说明
dataset  数据集
classVec 标签列
vocablist 词汇表
inputset  切分好的一个文档
returnVec 训练集向量
trainMat 训练集向量矩阵
p1v  侮辱类词条的条件概率数组
p0v  非侮辱类词条的条件概率数组
pA0  侮辱性文档占总文档的概率
2，函数作用
creatVocabList(dataset) 构建词汇表
setOfWords2Vec(vocablist,inputset) 获得训练集向量
get_trainMat(dataset) 获得训练集向量矩阵
trainNB(trainMat,classVec) 朴素贝叶斯分类器训练函数
classifyNB(vec2classify,p1v,p0v,pA0) 测试朴素贝叶斯分类器
testingNB(testVec) 朴素贝叶斯测试函数
trainNB2(trainMat,classVec)  朴素贝叶斯分类器训练函数改进版
classifyNB2(vec2classify,p1v,p0v,pA0) 测试朴素贝叶斯分类器改进版
testingNB2(testVec) 朴素贝叶斯测试函数改进版
