import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

"""
图像增强是指按特定的需要突出一幅图像中的某些信息，同时，消弱或去除某些不需要的信息的处理方法。
其主要目的是处理后的图像对某些特定的应用比原来的图像更加有效。
图像增强技术主要有直方图修改处理、图像平滑化处理、图像尖锐化处理和彩色处理技术等。
"""


# 灰度直方图
# 直方图是多种空间域处理技术的基础。直方图操作能有效地用于图像增强。
# 除了提供有用的图像统计资料外，直方图固有的信息在其他图像处理应用中也是非常有用的，如图像压缩与分割。
# 直方图在软件中易于计算，也适用于商用硬件设备，因此，它们成为了实时图像处理的一个流行工具。
# 直方图是图像的最基本的统计特征，它反映的是图像的灰度值的分布情况。
# 直方图均衡化的目的是使图像在整个灰度值动态变化范围内的分布均匀化，改善图像的亮度分布状态，增强图像的视觉效果。
# 灰度直方图是图像预处理中涉及最广泛的基本概念之一。
# 图像的直方图事实上就是图像的亮度分布的概率密度函数，是一幅图像的所有象素集合的最基本的统计规律。
# 直方图反映了图像的明暗分布规律，可以通过图像变换进行直方图调整，获得较好的视觉效果。
# 直方图均衡化是通过灰度变换将一幅图像转换为另一幅具有均衡直方图，即在每个灰度级上都具有相同的象素点数的过程。
def img_hist_function(img):
    img_GRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", img_GRAY)
    cv.waitKey(0)
    # 绘制直方图
    hist_ori = cv.calcHist([img_GRAY], [0], None, [256], [0, 255])
    plt.figure()
    plt.plot(hist_ori)

    # 直方图均衡化
    equ = cv.equalizeHist(img_GRAY)
    cv.imshow("equ", equ)
    cv.waitKey(0)
    cv.destroyAllWindows()
    hist_equ = cv.calcHist([equ], [0], None, [256], [0, 255])
    plt.figure()
    plt.plot(hist_equ)
    plt.show()

# 滤波也可对RGB的每个通道单独进行滤波，然后再合并通道，这样可以保留图像的颜色信息
# 空域中的平滑滤波
# 平滑滤波是低频增强的空间域滤波技术。它的目的有两类：一类是模糊；另一类是消除噪音。
# 空间域的平滑滤波一般采用简单平均法进行，就是求邻近像元点的平均亮度值。
# 邻域的大小与平滑的效果直接相关，邻域越大平滑的效果越好。
# 但邻域过大，平滑会使边缘信息损失的越大，从而使输出的图像变得模糊，因此需合理选择邻域的大小。
def img_smoothF_function(img):
    img_GRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 均值滤波，卷积核大小为 3*5, 卷积核的大小是可以设定的
    blur_mean = cv.blur(img_GRAY, (3, 5))
    box_mean = cv.boxFilter(img_GRAY, -1, (3, 5))
    cv.imshow("blur_mean", blur_mean)
    cv.waitKey(0)
    cv.imshow("box_mean", box_mean)
    cv.waitKey(0)

    # 高斯模糊滤波，(5,5)表示的是卷积核的大小，0 表示的是沿 x 与 y 方向上的标准差
    blur_gauss = cv.GaussianBlur(img_GRAY, (5, 5), 0)
    cv.imshow("blur_gauss", blur_gauss)
    cv.waitKey(0)

    # 中值滤波，5 表示的是卷积核的大小
    blur_mid = cv.medianBlur(img_GRAY, 5)
    cv.imshow("blur_mid", blur_mid)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 空域中的锐化滤波
# 图像锐化(image sharpening)是补偿图像的轮廓，增强图像的边缘及灰度跳变的部分，使图像变得清晰
def img_sharpF_function(img):
    # 灰度化
    img_GRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 图像高斯滤波去噪
    blur = cv.GaussianBlur(img_GRAY, (3, 3), 1, 1)  # 核尺寸通过对图像的调节自行定义
    # 图像阈值化处理
    ret, thresh = cv.threshold(blur, 127, 255, cv.THRESH_BINARY)  # 二进制阈值化
    # 此处还可以进行形态学处理，前面如果达标，这步骤可以省略

    # 轮廓提取

    # Roberts 算子
    # Roberts 算法又称为交叉微分算法，它是基于交叉差分的梯度算法，通过局部差分计算检测边缘线条。
    # 常用来处理具有陡峭的低噪声图像。
    # 当图像边缘接近于正 45 度或负 45 度时，该算法处理效果更理想。
    # 定义 Roberts 算子的卷积核
    R_kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    R_kernely = np.array([[0, -1], [1, 0]], dtype=int)
    # 调用 OpenCV 的 filter2D() 函数利用卷积核实现对图像的卷积运算，实现边缘提取
    R_x = cv.filter2D(thresh, cv.CV_16S, R_kernelx)
    R_y = cv.filter2D(thresh, cv.CV_16S, R_kernely)
    # 取绝对值转 uint8
    R_absX = cv.convertScaleAbs(R_x)
    R_absY = cv.convertScaleAbs(R_y)
    # 通过 addWeighted() 函数来进行 x 方向与 y 方向上的结合
    Roberts = cv.addWeighted(R_absX, 0.5, R_absY, 0.5, 0)
    cv.imshow("Roberts", Roberts)
    cv.waitKey()

    # Prewitt 算子
    # Prewitt 算子是一种一阶微分算子的边缘检测，利用像素点上下、左右邻点的灰度差，在边缘处达到极值检测边缘，去掉部分伪边缘，对噪声具有平滑作用。
    # 其原理是在图像空间利用两个方向模板与图像进行邻域卷积来完成的，这两个方向模板一个检测水平边缘，一个检测垂直边缘。
    # Prewitt 算法适合用来识别噪声较多、灰度渐变的图像；
    # 定义 Roberts 算子的卷积核
    P_kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    P_kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    # 调用 OpenCV 的 filter2D() 函数利用卷积核实现对图像的卷积运算，实现边缘提取
    P_x = cv.filter2D(thresh, -1, P_kernelx)
    P_y = cv.filter2D(thresh, -1, P_kernely)
    # 取绝对值转 uint8
    P_absX = cv.convertScaleAbs(P_x)
    P_absY = cv.convertScaleAbs(P_y)
    # 通过 addWeighted() 函数来进行 x 方向与 y 方向上的结合
    Prewitt = cv.addWeighted(P_absX, 0.5, P_absY, 0.5, 0)
    cv.imshow("Prewitt", Prewitt)
    cv.waitKey()

    # Sobel 算子
    # Sobel 算法(索贝尔算子)是一种用于边缘检测的离散微分算子，它结合了高斯平滑和微分求导。
    # 该算子用于计算图像明暗程度近似值，根据图像边缘旁边明暗程度把该区域内超过某个数的特定点记为边缘。
    # Sobel 算子在 Prewitt 算子的基础上增加了权重的概念，认为相邻点的距离远近对当前像素点的影响是不同的，
    # 距离越近的像素点对应当前像素的影响越大，从而实现图像锐化并突出边缘轮廓。
    # 当对精度要求不是很高时，Sobel 算子是一种较为常用的边缘检测方法。
    # Roberts 算子的卷积核
    # So_kerneldx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)
    # So_kerneldy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
    # OpenCV 中提供了专门的 Soble 算法的库函数,可以直接使用
    S_x = cv.Sobel(thresh, cv.CV_16S, 1, 0)  # 对 x 求一阶导
    S_y = cv.Sobel(thresh, cv.CV_16S, 0, 1)  # 对 y 求一阶导
    # 取绝对值转 uint8
    S_absX = cv.convertScaleAbs(S_x)
    S_absY = cv.convertScaleAbs(S_y)
    # 通过 addWeighted() 函数来进行 x 方向与 y 方向上的结合
    Sobel = cv.addWeighted(S_absX, 0.5, S_absY, 0.5, 0)
    cv.imshow("Sobel", Sobel)
    cv.waitKey()
    cv.destroyAllWindows()


# 频域中的低通（平滑）滤波器
def low_pass_filtering(img, size):  # size为滤波尺寸
    img_GRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 傅里叶变换
    img_dft = np.fft.fft2(img_GRAY)
    dft_shift = np.fft.fftshift(img_dft)  # 将频域从左上角移动到中间
    # 低通滤波
    h, w = dft_shift.shape[0:2]  # 获取图像属性（高、宽和图像通道数）
    h_center, w_center = int(h / 2), int(w / 2)  # 找到傅里叶频谱图的中心点
    dft_shift_black = np.zeros((h, w), np.uint8)
    # 中心点加减滤波尺寸的一半，刚好形成一个定义尺寸的滤波大小，然后设置为1，保留低频部分
    dft_shift_black[
        h_center - int(size / 2) : h_center + int(size / 2),
        w_center - int(size / 2) : w_center + int(size / 2),
    ] = 1
    dft_shift = dft_shift * dft_shift_black  # 将定义的低通滤波与传入的傅里叶频谱图一一对应相乘，得到低通滤波
    res = np.log(np.abs(dft_shift))
    # 傅里叶逆变换
    idft_shift = np.fft.ifftshift(dft_shift)  # 将频域从中间移动到左上角
    ifimg = np.fft.ifft2(idft_shift)
    ifimg = np.abs(ifimg)
    cv.imshow("lowPassFilter", np.int8(ifimg))
    cv.waitKey(0)
    cv.destroyAllWindows()


# 频域中的高通（锐化）滤波器
def high_pass_filtering(img, size):  # size为滤波尺寸
    img_GRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 傅里叶变换
    img_dft = np.fft.fft2(img_GRAY)
    dft_shift = np.fft.fftshift(img_dft)  # 将频域从左上角移动到中间
    # 高通滤波
    h, w = dft_shift.shape[0:2]  # 获取图像属性（高、宽和图像通道数）
    h_center, w_center = int(h / 2), int(w / 2)  # 找到傅里叶频谱图的中心点
    # 中心点加减滤波尺寸的一半，刚好形成一个定义尺寸的滤波大小，然后设置为0
    dft_shift[
        h_center - int(size / 2) : h_center + int(size / 2),
        w_center - int(size / 2) : w_center + int(size / 2),
    ] = 0
    res = np.log(np.abs(dft_shift))
    # 傅里叶逆变换
    idft_shift = np.fft.ifftshift(dft_shift)  # 将频域从中间移动到左上角
    img_idft = np.fft.ifft2(idft_shift)
    img_idft = np.abs(img_idft)
    cv.imshow("highPassFilter", np.int8(img_idft))
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # 读取图像，并以 numpy 数组形式储存
    img = cv.imread("Experiment1-4/2.jpg")
    img_hist_function(img)
    cv.destroyAllWindows()
    img_smoothF_function(img)
    img_sharpF_function(img)
    cv.destroyAllWindows()
    low_pass_filtering(img, 100)
    high_pass_filtering(img, 50)
    cv.destroyAllWindows()
