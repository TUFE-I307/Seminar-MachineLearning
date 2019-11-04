### Question 1 :步长的几何意义
  图像中每一段曲线的长度

---

### Question 2 :为什么用sigmoid函数
- sigmoid函数是一个阀值函数，不管x取什么值，对应的sigmoid函数值总是0<sigmoid(x)<1。
- sigmoid函数严格单调递增，而且其反函数也单调递增
- sigmoid函数连续
- sigmoid函数光滑
- sigmoid函数关于点(0, 0.5)对称
- sigmoid函数的导数是以它本身为因变量的函数，即f(x)' = F(f(x))

---

### Question 3:sigmoid函数推导过程
伯努利分布：

<html>
<a href="https://www.codecogs.com/eqnedit.php?latex=f(x|p)&space;=p^x{(1-p)}^{(1-x)}\\&space;=exp\{xln\frac{p}{1-p}&plus;ln(1-p)\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x|p)&space;=p^x{(1-p)}^{(1-x)}\\&space;=exp\{xln\frac{p}{1-p}&plus;ln(1-p)\}" title="f(x|p) =p^x{(1-p)}^{(1-x)}\\ =exp\{xln\frac{p}{1-p}+ln(1-p)\}" /></a>
</html>


sigmoid函数：

<html>
<a href="https://www.codecogs.com/eqnedit.php?latex=\\y(p)=ln\frac{p}{1-p}\\&space;e^{y(p)}=\frac{p}{1-p}\\&space;p=\frac{1}{1&plus;e^{-y(p)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\y(p)=ln\frac{p}{1-p}\\&space;e^{y(p)}=\frac{p}{1-p}\\&space;p=\frac{1}{1&plus;e^{-y(p)}}" title="\\y(p)=ln\frac{p}{1-p}\\ e^{y(p)}=\frac{p}{1-p}\\ p=\frac{1}{1+e^{-y(p)}}" /></a>
</html>

令 y(p)=x, p=f(x),则 
<html>
<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)=\frac{1}{1&plus;e^{-x}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{1&plus;e^{-x}}" title="f(x)=\frac{1}{1+e^{-x}}" /></a>
<html>
  
  
证毕。

---
如果有哪里写的不对，欢迎大家指正。——韩琳琳
