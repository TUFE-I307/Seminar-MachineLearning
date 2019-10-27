# 谱聚类(Spectral Clustering)  
+ 复杂网络与社交网络中的谱聚类算法是一类无监督社区发现算法,通过事先给定社区数目k,将网络的拉普拉斯矩阵(Laplacian Matrix)的前k个最小特征值对于的特征向量作为输入，利用K-MEANS算法进行聚类。  
+ 该部分仅介绍谱聚类算法的步骤与当社区数目为2时的算法有效性推导  
## 1.算法步骤  
**Algorithm : Spectral Clustering**  
**Input : 网络G=(V,E), 社区数目k**  
**Output : 社区划分**  
1.计算G的邻接矩阵A与度矩阵D；  
2.计算G的拉普拉斯矩阵L=D-A；  
3.计算规范化拉普拉斯矩阵![](http://latex.codecogs.com/gif.latex?L{}'=D^{-1/2}LD^{-1/2})；  
4.计算![](http://latex.codecogs.com/gif.latex?L{}')的特征值与特征向量, 并按特征值的大小排序；
5.选取Ls的前k个非0最小特征值的特征向量, 记为X=[v1, v2, ..., vk]；  
6.将X作为输入, 使用K-MEANS算法进行聚类；  
7.K-MEANS得到的k个类即为谱聚类算法的结果。  
## 2.谱聚类算法的有效性推导(k=2)
从上述算法步骤中可以看出，当社区数目为2时，谱聚类算法等价于利用规范化拉普拉斯矩阵![](http://latex.codecogs.com/gif.latex?L{}')的第二小特征值所对应的特征向量(**也称为费德勒向量，Fiedler vector**)中的元素符号来将节点进行分类。  
这里容易产生两个问题：  
**1.为什么要使用Fiedler vector？**  
**2.为什么使用Fiedler vector中的元素符号来划分节点社区(类别)是一个有效的方案？**  
首先，我们要对图论中的划分准则进行一定的说明，[这里](https://wenku.baidu.com/view/549bfe7a66ec102de2bd960590c69ec3d4bbdb46.html)给出了几种常见的划分准则，此处我们只以比例割集准则为例进行推导。  
首先，假设将网络(图)G=(V,E)分割为k个不相交的子图![](http://latex.codecogs.com/gif.latex?C_{1})、![](http://latex.codecogs.com/gif.latex?C_{2})...![](http://latex.codecogs.com/gif.latex?C_{k})，将这些子图称之为网络的**社区**(**Community**)。将连接不同社区节点的边称为**桥**(**bridge**)，将连接![](http://latex.codecogs.com/gif.latex?C_{i})和![](http://latex.codecogs.com/gif.latex?C_{j})的桥的数量记为  
![](http://latex.codecogs.com/gif.latex?W(C_{i},C_{j})=\sum_{i\in\C_{i},j\in\C_{j}}a_{ij})  
则G的桥的数量为  
![](http://latex.codecogs.com/gif.latex?bridge(C_{1},C_{2},...,C_{k})=\frac{1}{2}\sum_{i=1}^{k}W(C_{i},\bar{C_{i}}))  
其中![](http://latex.codecogs.com/gif.latex?\bar{C_{i}})称为![](http://latex.codecogs.com/gif.latex?C_{i})的补图，即
![](http://latex.codecogs.com/gif.latex?C_{i}\cup\bar{C_{i}}=G)  
由于此处仅考虑k=2的情况，则此时有![](http://latex.codecogs.com/gif.latex?bridge(C_{1},C_{2})=W(C_{1},C_{2}))。  
根据G的拉普拉斯矩阵L=D-A，对![](http://latex.codecogs.com/gif.latex?\forall\textbf{x}\in\textbf{R}^{n})有  
![](http://latex.codecogs.com/gif.latex?\textbf{x}^{T}\textbf{L}\textbf{x}=\textbf{x}^{T}(\textbf{D}-\textbf{A})\textbf{x}=\textbf{x}^{T}\textbf{D}\textbf{x}-\textbf{x}^{T}\textbf{A}\textbf{x}=\sum_{i=1}^{n}d_{i}x_{i}^{2}-\sum_{i,j=1}^{n}a_{ij}x_{i}x_{j})  
![](http://latex.codecogs.com/gif.latex?=\frac{1}{2}\left[\sum_{i=1}^{n}d_{i}x_{i}^{2}-2\sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij}x_{i}x_{j}+\sum_{i=1}^{n}d_{i}x_{i}^{2}\right])  
![](http://latex.codecogs.com/gif.latex?=\frac{1}{2}\left[\sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij}x_{i}^{2}-2\sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij}x_{i}x_{j}+\sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij}x_{j}^{2}\right])  
![](http://latex.codecogs.com/gif.latex?=\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij}(x_{i}-x_{j})^{2})  
此处我们假设网络G在实际中确实存在一个最优的二分方案，若用一个向量来表示该方案，则属于同一社区的节点在该向量中的值与符号应当相等。在比例割集准则下，我们不妨令该向量为列向量，且向量中的元素为  
若![](http://latex.codecogs.com/gif.latex?i\in)![](http://latex.codecogs.com/gif.latex?C_{1})，![](http://latex.codecogs.com/gif.latex?f_{i}=\sqrt{\frac{\left|C_{2}\right|}{\left|C_{1}\right|}})；若![](http://latex.codecogs.com/gif.latex?i\in)![](http://latex.codecogs.com/gif.latex?C_{2})，![](http://latex.codecogs.com/gif.latex?f_{i}=-\sqrt{\frac{\left|C_{1}\right|}{\left|C_{2}\right|}})。  
代入上式，则有  
![](http://latex.codecogs.com/gif.latex?f^{T}\textbf{L}f=\frac{1}{2}\sum_{i,j=1}^{n}a_{ij}(f_{i}-f_{j})^{2})  
![](http://latex.codecogs.com/gif.latex?=\frac{1}{2}\sum_{C_{1},C_{2}}a_{ij}(f_{i}-f_{j})^{2}+\frac{1}{2}\sum_{C_{2},C{1}}a_{ij}(f_{i}-f_{j})^{2})  
![](http://latex.codecogs.com/gif.latex?=\frac{1}{2}\sum_{C_{1},C_{2}}a_{ij}\left(\sqrt{\frac{\left|C_{2}\right|}{\left|C_{1}\right|}}+\sqrt{\frac{\left|C_{1}\right|}{\left|C_{2}\right|}}\right)^{2}+\frac{1}{2}\sum_{C_{2},C{1}}a_{ij}\left(-\sqrt{\frac{\left|C_{1}\right|}{\left|C_{2}\right|}}-\sqrt{\frac{\left|C_{2}\right|}{\left|C_{1}\right|}}\right)^{2})  
![](http://latex.codecogs.com/gif.latex?=\frac{1}{2}\sum_{C_{1},C_{2}}a_{ij}\left(\frac{\left|C_{2}\right|}{\left|C_{1}\right|}+\frac{\left|C_{1}\right|}{\left|C_{2}\right|}+2\right))  
![](http://latex.codecogs.com/gif.latex?=\frac{1}{2}\sum_{C_{1},C_{2}}a_{ij}\left(\frac{\left|C_{2}\right|+\left|C_{1}\right|}{\left|C_{1}\right|}+\frac{\left|C_{1}\right|+\left|C_{2}\right|}{\left|C_{2}\right|}\right))  
![](http://latex.codecogs.com/gif.latex?=\left(\left|C_{1}\right|+\left|C_{2}\right|\right)\left(\frac{1}{\left|C_{1}\right|}+\frac{1}{\left|C_{2}\right|}\right)\sum_{C_{1},C_{2}}a_{ij})  
![](http://latex.codecogs.com/gif.latex?=n\left(\frac{1}{\left|C_{1}\right|}+\frac{1}{\left|C_{2}\right|}\right)bridge(C_{1},C_{2}))  
其中![](http://latex.codecogs.com/gif.latex?\sum_{C_{1},C_{2}}a_{ij})表示i在![](http://latex.codecogs.com/gif.latex?C_{1})中、j在![](http://latex.codecogs.com/gif.latex?C_{2})中。应当注意的是，![](http://latex.codecogs.com/gif.latex?f)具有两个性质  
+ ![](http://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}f_{i}=\sum_{C_{1}}f_{i}+\sum_{C_{2}}f_{i}=\left|C_{1}\right|\cdot\sqrt{\frac{\left|C_{2}\right|}{\left|C_{1}\right|}}-\left|C_{2}\right|\cdot\sqrt{\frac{\left|C_{2}\right|}{\left|C_{1}\right|}}=0)  
+ ![](http://latex.codecogs.com/gif.latex?\left||f\right||_{2}=\sum_{i=1}^{n}f_{i}^{2}=\sum_{C_{1}}\frac{\left|C_{2}\right|}{\left|C_{1}\right|}+\sum_{C_{2}}\frac{\left|C_{1}\right|}{\left|C_{2}\right|}=\left|C_{2}\right|+\left|C_{1}\right|=n)  
即![](http://latex.codecogs.com/gif.latex?f^{T}\cdot\textbf{1}=0)且![](http://latex.codecogs.com/gif.latex?\left||f\right||_{2}=n)
