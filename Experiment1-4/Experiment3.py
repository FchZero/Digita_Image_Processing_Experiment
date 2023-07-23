import cv2 as cv
import numpy as np

"""
数学形态学是以形态结构元素为基础对图像进行分析的数学工具。
它的基本思想是用具有一定形态的结构元素去度量和提取图像中的对应形状以达到对图像分析和识别的目的。
数学形态学的应用可以简化图像数据，保持它们基本的形状特征，并除去不相干的结构。
数学形态学的基本运算有 4 个：膨胀、腐蚀、开启和闭合。它们在二值图像中和灰度图像中各有特点。
基于这些基本运算还可以推导和组合成各种数学形态学实用算法。
基本的形态运算是腐蚀和膨胀。
在形态学中，结构元素是最重要最基本的概念。结构元素在形态变换中的作用相当于信号处理中的“滤波窗口”。
用 B（x）代表结构元素，对工作空间 E 中的每一点 x，腐蚀和膨胀的定义为：
腐蚀：X = E ⊙ B(x)；
膨胀：Y = E ⊗ B(y)。
用 B（x）对 E 进行膨胀的结果就是把结构元素 B 平移后使 B 与 E 的交集非空的点构成的集合。
先腐蚀后膨胀的过程称为开运算。它具有消除细小物体，在纤细处分离物体和平滑较大物体边界的作用。
先膨胀后腐蚀的过程称为闭运算。它具有填充物体内细小空洞，连接邻近物体和平滑边界的作用。
可见，二值形态膨胀与腐蚀可转化为集合的逻辑运算，算法简单，适于并行处理，且易于硬件实现，适于对二值图像进行图像分割、细化、抽取骨架、边缘提取、形状分析。
但是，在不同的应用场合，结构元素的选择及其相应的处理算法是不一样的，对不同的目标图像需设计不同的结构元素和不同的处理算法。
结构元素的大小、形状选择合适与否，将直接影响图像的形态运算结果。
"""


# 灰度直方图
def img_Mor_function():
    img = cv.imread("Experiment1-4/3.jpg")
    # 灰度化
    img_GRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 二值化
    ret, thresh = cv.threshold(img_GRAY, 127, 255, cv.THRESH_BINARY)
    # 卷积核为3*3的全1矩阵
    kernel = np.ones((3, 3), np.uint8)
    # 腐蚀
    erosion = cv.erode(thresh, kernel)
    cv.imshow("erosion", erosion)
    cv.waitKey(0)
    # 膨胀
    dilation = cv.dilate(thresh, kernel, iterations=1)
    cv.imshow("dilation", dilation)
    cv.waitKey(0)
    # 开运算（先腐蚀再膨胀）
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    cv.imshow("opening", opening)
    cv.waitKey(0)
    # 闭运算（先膨胀再腐蚀）
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    cv.imshow("closing", closing)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    img_Mor_function()
