import cv2 as cv


def img_basic_function():
    # 使用imread(filename, flag)函数读取图像
    # flag取值0为原图像的灰度图；取值1为BGR彩图；取值2为原深度图像；取值4为原颜色图像
    # 读取的图像以numpy中的三维数组的形式存储
    # 第1个维度即最外面的一维"[[[]]]"表示整个图像各个像素的BGR值(3维数组[[[B G R][B G R]...[B G R]] [[B G R][B G R]...[B G R]] ... [[B G R][B G R]...[B G R]]])
    # 第2个维度即中间维度"[[]]"表示某行各个像素的BGR值（2维数组[[B G R] [B G R] ... [B G R]]])
    # 第3个维度即最里面的一维"[]"表示某个像素的BGR值（1维数组[B G R]）
    # img[0]代表第1行所有像素的BGR值，img[0][0]代表第1行第1个像素的BGR值，img[0][0][0]代表第1行第1个像素的B值
    img = cv.imread("Experiment1-4/1.jpg")
    # .shape[]读取矩阵各个维度的长度，.shape[i]是读取矩阵第i + 1个维度的长度
    # 所以img.shape返回的元组（tuple）中的三个数就依次表示行数（高度）、列数（宽度）和通道数
    # tuple是一个不可变的序列，一旦创建就不能修改。它可以包含不同类型的元素，并使用逗号分隔。例如，(1, 'hello', 3.14)就是一个Tuple
    # list是一个可变的序列，可以随意添加、修改和删除其中的元素。它也可以包含不同类型的元素，并使用方括号[]来表示。例如，[1, 'hello', 3.14]就是一个List。
    # array是一个多维数组，可以用来表示矩阵、向量等数学对象。它需要导入Numpy库，并使用np.array()函数来创建。例如，np.array([[1, 2], [3, 4]])就是一个2x2的矩阵。
    print(img.shape)
    # 查看图片的类型，常用uint8表示，代表一个无符号的8位整数
    print(img.dtype)
    # 使用imshow函数显示图像，第一个参数是窗口名称（可不写），第二个参数是要显示的图像的名称，一定要写
    cv.imshow("img", img)
    # 可以让窗口一直显示图像直到按下任意按键
    cv.waitKey(0)

    # 使用cv.cvtColor()函数转换色彩空间
    # 参数‘cv.COLOR_BGR2GRAY’表示从RGB空间转换到灰度空间
    # 灰度空间中的图像以numpy中的二维数组的形式存储
    # 第一个维度即外面的维度"[[]]"表示整个图像各个像素的灰度值（[[Gray Gray ... Gray] [Gray Gray ... Gray] ... [Gray Gray ... Gray]]）
    # 第二个维度即里面的维度"[]"表示某行各个像素的灰度值（一维数组[Gray Gray ... Gray]）
    # img_GRAY[0]代表第一行所有像素的灰度值，img_GRAY[0][0]代表第一行第一个像素的灰度值
    img_GRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", img_GRAY)
    cv.waitKey(0)

    # 使用 cv.threshold函数进行图像阈值处理。
    # 参数‘cv.THRESH_BINARY’代表了阈值的类型，127为阈值。
    # 返回值中ret为阈值，阈值是用于将输入图像分割成两个部分的像素值。它可以是手动指定的固定值，也可以是根据图像的直方图自动计算得出的。
    # thresh为处理后的图像。
    ret, thresh = cv.threshold(img_GRAY, 127, 255, cv.THRESH_BINARY)
    cv.imshow("thresh", thresh)
    cv.waitKey(0)

    # 使用 cv.resize 函数进行图像缩放
    res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    cv.imshow("resize", res)
    cv.waitKey(0)

    # 0表示沿x轴翻转，正表示沿y轴翻转，负表示沿x和y轴翻转
    flip = cv.flip(img, 1)
    cv.imshow("flip", flip)
    cv.waitKey(0)
    # 销毁所有窗口
    cv.destroyAllWindows()
    # 保存图像
    # cv.imwrite("result.jpg", res)


if __name__ == "__main__":
    img_basic_function()
