from numpy import inf
from numpy import zeros
import numpy as np
from sklearn.model_selection import train_test_split


# 生成数据集。数据集包括标签，全包含在返回值的dataset上
def get_Datasets():
    from sklearn.datasets import make_classification
    dataSet, classLabels = make_classification(n_samples=200, n_features=100, n_classes=2)
    return np.concatenate((dataSet, classLabels.reshape((-1, 1))), axis=1)
    print(dataSet.shape, classLables.shape)

####################步骤一##############################
# 构建n个子集   bootstrap
def get_subsamples(dataSet, n):
    subDataSet = []
    for i in range(n):
        index = []  # 每次都重新选择k个 索引
        for k in range(len(dataSet)):  # 长度是k
            index.append(np.random.randint(len(dataSet)))  # (0,len(dataSet)) 内的一个整数
        subDataSet.append(dataSet[index, :])
    return subDataSet
############################步骤二、三################################
# 根据某个特征及值对数据进行分类
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] < value)[0], :]
    return mat0, mat1


# 计算方差，回归时使用
def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


# 计算平均值，回归时使用
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

#将分类结果最多的分类作为最终结果
def MostNumber(dataSet):
    # number=set(dataSet[:,-1])
    #np.nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数。
    len0 = len(np.nonzero(dataSet[:, -1] == 0)[0])
    len1 = len(np.nonzero(dataSet[:, -1] == 1)[0])
    if len0 > len1:
        return 0
    else:
        return 1 


# 计算基尼指数   一个随机选中的样本在子集中被分错的可能性   是被选中的概率乘以被分错的概率
def gini(dataSet):
    corr = 0.0
    for i in set(dataSet[:, -1]):  # i 是这个特征下的 某个特征值
        corr += (len(np.nonzero(dataSet[:, -1] == i)[0]) / len(dataSet)) ** 2
    return 1 - corr

#在每个节点进行分类时选择一个最优的属性进行分裂
def select_best_feature(dataSet, m, alpha="huigui"):
    f = dataSet.shape[1]  # 看这个数据集有多少个特征，即f个
    index = []
    bestS = inf;
    bestfeature = 0;
    bestValue = 0;
    if alpha == "huigui":
        S = regErr(dataSet)
    else:
        S = gini(dataSet)

    for i in range(m):
        index.append(np.random.randint(f))  # 在f个特征里随机选择m个特征，然后在这m个特征里选择一个合适的分类特征。

    for feature in index:
        for splitVal in set(dataSet[:, feature]):  # set() 函数创建一个无序不重复元素集，用于遍历这个特征下所有的值
            mat0, mat1 = binSplitDataSet(dataSet, feature, splitVal)
            if alpha == "huigui":
                newS = regErr(mat0) + regErr(mat1)  # 计算每个分支的回归方差
            else:
                newS = gini(mat0) + gini(mat1)  # 计算基尼系数
            if bestS > newS:
                bestfeature = feature
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < 0.001 and alpha == "huigui":  # 对于回归来说，方差足够小了，那就取这个分支的均值
        return None, regLeaf(dataSet)
    elif (S - bestS) < 0.001:
        return None, MostNumber(dataSet)  # 对于分类来说，被分错率足够小了，那这个分支的分类就是大多数所在的类。
    # mat0,mat1=binSplitDataSet(dataSet,feature,splitVal)
    return bestfeature, bestValue

# 实现决策树，使用20个特征，深度为10，
def createTree(dataSet, alpha="huigui", m=20, max_level=10): 
    bestfeature, bestValue = select_best_feature(dataSet, m, alpha=alpha)
    if bestfeature == None:
        return bestValue
    retTree = {}
    max_level -= 1
    if max_level < 0:  # 控制深度
        return regLeaf(dataSet)
    retTree['bestFeature'] = bestfeature
    retTree['bestVal'] = bestValue
    lSet, rSet = binSplitDataSet(dataSet, bestfeature,bestValue)  
    # lSet是根据特征bestfeature分到左边的向量，rSet是根据特征bestfeature分到右边的向量
    retTree['right'] = createTree(rSet, alpha, m, max_level)
    retTree['left'] = createTree(lSet, alpha, m, max_level)  # 每棵树都是二叉树，往下分类都是一分为二。
    return retTree
#########################步骤四#################################
#随机森林
def RondomForest(dataSet, n, alpha="huigui"):  
    Trees = []  # 设置一个空树集合
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(dataSet[:, :-1], dataSet[:, -1], test_size=0.33,random_state=42)
        X_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
        Trees.append(createTree(X_train, alpha=alpha))
    return Trees  # 生成很多树


###################################################################

# 预测单个数据样本 如何利用已经训练好的一棵树对单个样本进行 回归或分类
def treeForecast(trees, data, alpha="huigui"):
    if alpha == "huigui":
        if not isinstance(trees, dict):  # isinstance() 函数来判断一个对象是否是一个已知的类型
            return float(trees)

        if data[trees['bestFeature']] > trees['bestVal']:  # 如果数据的这个特征大于阈值，那就调用左支
            if type(trees['left']) == 'float':  # 如果左支已经是节点了，就返回数值。如果左支还是字典结构，那就继续调用， 用此支的特征和特征值进行选支。
                return trees['left']
            else:
                return treeForecast(trees['left'], data, alpha)
        else:
            if type(trees['right']) == 'float':
                return trees['right']
            else:
                return treeForecast(trees['right'], data, alpha)
    else:
        if not isinstance(trees, dict):  # 分类和回归是同一道理
            return int(trees)

        if data[trees['bestFeature']] > trees['bestVal']:
            if type(trees['left']) == 'int':
                return trees['left']
            else:
                return treeForecast(trees['left'], data, alpha)
        else:
            if type(trees['right']) == 'int':
                return trees['right']
            else:
                return treeForecast(trees['right'], data, alpha)

#对data集合进行预测 调用treeForecast加循环
def createForeCast(trees, test_dataSet, alpha="huigui"):
    cm = len(test_dataSet)
    yhat = np.mat(zeros((cm, 1)))
    for i in range(cm):  
        yhat[i, 0] = treeForecast(trees, test_dataSet[i, :], alpha)  #
    return yhat


# 随机森林预测
def predictTree(Trees, test_dataSet, alpha="huigui"):  # Trees 是已经训练好的随机森林   
    cm = len(test_dataSet)
    yhat = np.mat(zeros((cm, 1)))
    for trees in Trees:
        yhat += createForeCast(trees, test_dataSet, alpha)  # 把每次的预测结果相加
    if alpha == "huigui":
        yhat /= len(Trees)  # 如果是回归的话，每棵树的结果应该是回归值，相加后取平均
    else:
        for i in range(len(yhat)):  # 如果是分类的话，每棵树的结果是一个投票向量，相加后，
                                    # 看每类的投票是否超过半数，超过半数就确定为1
            if yhat[i, 0] > len(Trees) / 2:
                yhat[i, 0] = 1
            else:
                yhat[i, 0] = 0
    return yhat

#####调用上述函数######
if __name__ == '__main__':
    dataSet = get_Datasets()
    print(dataSet[:, -1].T)  # 打印标签，与后面预测值对比  .T其实就是对一个矩阵的转置
    RomdomTrees = RondomForest(dataSet, 4, alpha="fenlei")  # 这里训练好了很多树的集合，就组成了随机森林。一会一棵一棵的调用。
    print("---------------------RomdomTrees------------------------")
    test_dataSet = dataSet  # 得到数据集和标签
    yhat = predictTree(RomdomTrees, test_dataSet, alpha="fenlei")  # 调用训练好的那些树。综合结果，得到预测值。
    print(yhat.T)
    print(dataSet[:, -1].T - yhat.T)

#原文链接：https: // blog.csdn.net / qq_40514570 / article / details / 92720952