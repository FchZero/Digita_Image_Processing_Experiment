import cv2 as cv


def img_ED_function():
    img = cv.imread("Experiment1-4/4.jpg")
    # 灰度化
    img_GRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 高斯滤波
    Gauss_blur = cv.GaussianBlur(img_GRAY, (3, 3), 1, 1)
    # LOG 检测器
    # 调用 Laplacian() 算法进行图像轮廓提取
    temp_LOG = cv.Laplacian(Gauss_blur, cv.CV_16S, ksize=1)
    LOG = cv.convertScaleAbs(temp_LOG)  # 得到 LOG 算法处理结果
    cv.imshow("LOG", LOG)
    cv.waitKey()

    # Scharr 算子
    # Scharr 算子又称为 Scharr 滤波器，也是计算 x 或 y 方向上的图像差分，在 OpenCV 中主要是配合 Sobel 算子的运算而存在的
    # Roberts 算子的卷积核
    # Sc_kerneldx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=int)
    # Sc_kerneldy = np.array([[-3, -10, 3], [0, 0, 0], [3, 10, 3]], dtype=int)
    # OpenCV 中提供了专门的 Scharr 算法的库函数,可以直接使用
    Scharr_x = cv.Scharr(Gauss_blur, cv.CV_16S, 1, 0)
    Scharr_y = cv.Scharr(Gauss_blur, cv.CV_16S, 0, 1)
    Scharr_absx = cv.convertScaleAbs(Scharr_x)
    Scharr_absy = cv.convertScaleAbs(Scharr_y)
    Scharr = cv.addWeighted(Scharr_absx, 0.5, Scharr_absy, 0.5, 0)
    cv.imshow("Scharr", Scharr)
    cv.waitKey(0)

    # Canny 检测器
    # Canny 边缘检测器是函数 edge 中最强大的边缘检测器。方法总结如下：
    # 1.使用具有指定标准差 δ 的一个高斯滤波器来平滑图像，以减少噪声;
    # 2.在每个点处计算局部梯度和边缘方向。边缘点定义为梯度方向强度局部最大的点;
    # 3.步骤 2 中确定的边缘点产生梯度中的脊线。
    # 然后，算法沿这些脊线的顶部进行追踪，并将实际上不在脊线顶部的像素设置为零，从而在输出中给出一条细线，该过程称为非最大值抑制。
    # 然后使用称为滞后阈值处理的方法来对这些脊线像素进行阈值处理，这一处理方法使用两个阈值T1和T2。其中T1 < T2。
    # 其值大于T2的脊线像素称为“强”边缘像素，值在T1和T2之间的脊线像素称为“弱”边缘像素。
    Gauss_blur = cv.GaussianBlur(img_GRAY, (7, 7), 1, 1)
    # Canny 算子进行边缘提取
    # T1 = 50, T2 = 150
    temp_Canny = cv.Canny(Gauss_blur, 50, 150)
    Canny = cv.convertScaleAbs(temp_Canny)
    cv.imshow("Canny", Canny)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    img_ED_function()
