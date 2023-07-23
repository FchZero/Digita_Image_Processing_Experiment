import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 展示函数
def display(name, image):
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def hist_equ(image):
    # 绘制直方图
    hist_ori = cv.calcHist([image], [0], None, [256], [0, 255])
    plt.figure()
    plt.plot(hist_ori)

    # 进行直方图均衡化。
    equ = cv.equalizeHist(image)
    display("equ", equ)
    hist_equ = cv.calcHist([equ], [0], None, [256], [0, 255])
    plt.figure()
    plt.plot(hist_equ)
    # plt.show()
    return equ


# 车牌定位函数
def location(image):
    image_b = cv.split(image)[0]
    image_g = cv.split(image)[1]
    image_r = cv.split(image)[2]

    # 彩色信息特征初步定位：车牌定位并给resize后的图像二值化赋值
    standard_b = 138
    standard_g = 63
    standard_r = 23
    standard_threshold = 50
    image_binary = img
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 提取与给定的r、g、b阈值相差不大的点(赋值为全白)
            if (
                abs(image_b[i, j] - standard_b) < standard_threshold
                and abs(image_g[i, j] - standard_g) < standard_threshold
                and abs(image_r[i, j] - standard_r) < standard_threshold
            ):
                image_binary[i, j, :] = 255
            # 其他所有的点赋值为全黑
            else:
                image_binary[i, j, :] = 0

    # 基于数学形态学进一步精细定位车牌区域
    # 灰度图
    img_edged = cv.Canny(image_binary, 30, 200)
    display("img_edged", img_edged)
    # 闭运算（先膨胀再腐蚀）
    kernel = np.ones((3, 3), np.uint8)
    image_closing = cv.morphologyEx(img_edged, cv.MORPH_CLOSE, kernel)
    # img_resize_dilate = cv.dilate(image, kernel, iterations = 5)  # 膨胀操作
    # img_resize_erosion = cv.erode(img_resize_dilate, kernel, iterations = 5)  # 腐蚀操作
    display("image_closing", image_closing)

    # 进行轮廓检索
    # findContours()在python3里返回2或3个值：image（可能没有）, contours, hierarchy
    # image：可能是跟输入contour类似的一张二值图像，也可能是原图像
    # contours：以list2维数组的形式存储
    # 第1个维度即最外面的一维"[[[]]]"表示所有轮廓各个像素的坐标(3维数组[[[x y][x y]...[x y]] [[x y][x y]...[x y]] ... [[x y][x y]...[x y]]])
    # 第2个维度即中间维度"[[]]"表示某个轮廓的各个像素的坐标（2维数组[[x y] [x y] ... [x y]]])
    # 第3个维度即最里面的一维"[]"表示某个像素的坐标（1维数组[x y]）
    # 所以通过contours[i]就可以对第i个轮廓进行操作，比如计算第i个轮廓的面积cv2.contourArea(contours[i])，或者画出第i个轮廓cv2.drawContours(img, contours[i], -1, (0, 0, 255), 3)
    # 注意：如果输入选择cv2.CHAIN_APPROX_SIMPLE，则同一个轮廓中的各个像素之间应该用直线连接起来，可以用cv2.drawContours()函数观察一下效果
    # hierarchy：以list2维数组的形式存储
    # 第1个维度即最外面的一维"[[[]]]"表示所有轮廓的层次结构(3维数组[[[a b c d][a b c d]...[a b c d]] [[a b c d][a b c d]...[a b c d]] ... [[a b c d][a b c d]...[a b c d]]])
    # 第2个维度即中间维度"[[]]"表示某个轮廓的层次结构（2维数组[[a b c d] [a b c d] ... [a b c d]]])
    # 第3个维度即最里面的一维"[]"表示层次结构（1维数组[a b c d]）
    # a、b、c、d分别对应下一个轮廓编号、上一个轮廓编号、父轮廓编号、子轮廓编号，某值为负数时表示没有对应项
    # 如果输入选择cv2.RETR_TREE则以树形结构组织输出
    contours = cv.findContours(img_edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    # 通过区域面积，宽高比例的方式进一步筛选车牌区域
    MIN_AREA = 400  # 设定矩形的最小区域，用于去除无用的噪声点
    license_contours = []
    for cnt in contours:
        # 框选生成最小外接矩形返回值（中心(x,y), (宽,高), 旋转角度）rect[0]：矩形中心点坐标；rect[1]：矩形的高和宽；rect[2]：矩形的旋转角度
        rect = cv.minAreaRect(cnt)
        area_width, area_height = rect[1]
        # 计算最小矩形的面积，初步筛选
        area = rect[1][0] * rect[1][1]  # 最小矩形面积
        if area > MIN_AREA:
            if area_width < area_height:  # 选择宽小于高的区域进行宽和高的置换
                area_width, area_height = area_height, area_width
            # 求出宽高之比(要求矩形区域长宽比在2到5.5之间，其他的排除)
            wh_ratio = area_width / area_height
            if 2 < wh_ratio < 5.5:
                license_contours.append(rect)
                box = cv.boxPoints(rect)  # 存放最小矩形的四个顶点坐标(先列后行的顺序)
                box = np.int0(box)  # 保留整数部分
    region = box
    return region


# 车牌分割函数
def segmentation(image):
    temp_col_index = []  # 存储含有字符的列的索引
    for col in range(image.shape[1]):
        if np.sum(image[:, col]) >= 5 * 255:  # 存在大于等于5个255的列存在字符
            temp_col_index.append(col)
    temp_col_index = np.array(temp_col_index)

    flag = 0  # 每个字符的起始列索引
    flag_i = 0  # 第flag_i个字符
    # 二维数组，每行代表每个字符的列索引，每行的列数为30，不足30的补0
    char_region_col = np.uint8(np.zeros([7, 30]))
    for j in range(temp_col_index.shape[0] - 1):
        # 相邻两个列索引的差值大于等于2且与上一个字符的起始列索引相差10个像素以上，说明这两个列索引之间没有字符
        if (temp_col_index[j + 1] - temp_col_index[j]) >= 2 and (temp_col_index[j + 1] - temp_col_index[flag]) >= 10:
            temp = temp_col_index[flag : j + 1]  # flag~j + 1为第flag_i个字符的列索引
            temp = np.append(temp, np.zeros(30 - temp.shape[0]))  # 补成30维的向量
            temp = np.uint8(temp.reshape(1, 30))
            char_region_col[flag_i, :] = temp
            flag = j + 1  # 下一个字符的起始列索引
            flag_i = flag_i + 1  # 下一个字符
    if char_region_col[6][0] == 0:  # 最后一个字符的列索引为0，说明最后一个字符没有被提取出来:
        temp = temp_col_index[flag:]
        temp = np.append(temp, np.zeros(30 - temp.shape[0]))  # 补成30维的向量
        temp = np.uint8(temp.reshape(1, 30))
        char_region_col[flag_i, :] = temp
    return char_region_col


# 字符提取函数
# 针对单个字符，用于去除其周围的边缘，并resize为25*15的图片：height=25,width=15
def char_extraction(image):
    # 提取满足条件(有2个以上的255的单元格)的列索引
    col_index = []
    for col in range(image.shape[1]):
        if np.sum(image[:, col]) >= 2 * 255:
            col_index.append(col)
    col_index = np.array(col_index)
    # 提取满足条件(有2个以上的255的单元格)的行索引
    row_index = []
    for row in range(image.shape[0]):
        if np.sum(image[row, :]) >= 2 * 255:
            row_index.append(row)
    row_index = np.array(row_index)
    # 按索引提取字符(符合条件的行列中取min~max)，并resize到25 * 15大小
    char_image = image[
        np.min(row_index) : np.max(row_index) + 1,
        np.min(col_index) : np.max(col_index) + 1,
    ]
    char_image = np.uint8(char_image)
    # 提取到的含有字符的列之间不是相邻的
    if col_index.shape[0] <= 2 or row_index.shape[0] <= 2:
        char_image = image[
            np.min(row_index) : np.max(row_index) + 1,
            np.min(col_index) : np.max(col_index) + 1,
        ]
        pad_row1 = np.int8(np.floor((25 - char_image.shape[0]) / 2))
        pad_row2 = np.int8(np.ceil((25 - char_image.shape[0]) / 2))
        pad_col1 = np.int8(np.floor((15 - char_image.shape[1]) / 2))
        pad_col2 = np.int8(np.ceil((15 - char_image.shape[1]) / 2))
        # cv.pad()用于在图像周围添加边框，参数为(图像,((上,下),(左,右)),填充方式,填充值)
        char_image = np.pad(
            char_image,
            ((pad_row1, pad_row2), (pad_col1, pad_col2)),
            "constant",
            constant_values=(0, 0),
        )
        char_image = np.uint8(char_image)
    else:
        char_image = cv.resize(char_image, (15, 25), interpolation=0)
    return char_image


# 模板生成函数
def template_generation(template_path, template_size):
    template_image_out = np.zeros([template_size, 25, 15], dtype=np.uint8)
    index = 0
    files = os.listdir(template_path)
    for file in files:
        template_image = cv.imdecode(np.fromfile(template_path + "/" + file, dtype=np.uint8), -1)
        template_image_gray = cv.cvtColor(template_image, cv.COLOR_BGR2GRAY)
        template_image_binary = cv.threshold(template_image_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        template_image_binary = 255 - template_image_binary  # 模板给出的与车牌上的是相反的,所有用255相减进行匹配
        template_image_out[index, :, :] = char_extraction(template_image_binary)
        index = index + 1
    return template_image_out


# 字符识别函数
def char_recognition():
    car_character = np.uint8(np.zeros([7, 25, 15]))
    car_character[0, :, :] = license_province1.copy()
    car_character[1, :, :] = license_province2.copy()
    car_character[2, :, :] = license_number1.copy()
    car_character[3, :, :] = license_number2.copy()
    car_character[4, :, :] = license_number3.copy()
    car_character[5, :, :] = license_number4.copy()
    car_character[6, :, :] = license_number5.copy()
    match_length = Chinese_char_template.shape[0] + Alphabet_char_template.shape[0] + Number_char_template.shape[0]
    match_mark = np.zeros([7, match_length])
    Chinese_char_start = 0
    Chinese_char_end = Chinese_char_template.shape[0]
    Alphabet_char_start = Chinese_char_template.shape[0]
    Alphabet_char_end = Chinese_char_template.shape[0] + Alphabet_char_template.shape[0]
    Number_char_start = Chinese_char_template.shape[0] + Alphabet_char_template.shape[0]
    Number_char_end = match_length
    for i in range(match_mark.shape[0]):  # 7个需识别的字符
        # 所有的汉字模板
        for j in range(Chinese_char_start, Chinese_char_end):
            match_mark[i, j] = cv.matchTemplate(car_character[i, :, :], Chinese_char_template[j, :, :], cv.TM_CCOEFF)
        # 所有的字母模板
        for j in range(Alphabet_char_start, Alphabet_char_end):
            match_mark[i, j] = cv.matchTemplate(
                car_character[i, :, :],
                Alphabet_char_template[j - Alphabet_char_start, :, :],
                cv.TM_CCOEFF,
            )
        # 所有的数字模板
        for j in range(Number_char_start, Number_char_end):
            match_mark[i, j] = cv.matchTemplate(
                car_character[i, :, :],
                Number_char_template[j - Number_char_start, :, :],
                cv.TM_CCOEFF,
            )
    license_index = np.argmax(match_mark, axis=1)
    license = []
    for i in range(license_index.shape[0]):
        if 0 <= license_index[i] <= 28:
            license.append(Chinese_character[license_index[i]])
        if 29 <= license_index[i] <= 54:
            license.append(Alphabet_character[license_index[i] - 29])
        if 55 <= license_index[i] <= 64:
            license.append(Number_character[license_index[i] - 55])
    # 打印识别结果
    # print(license)
    print("车牌号为：", end="")
    for i in range(len(license)):
        print(license[i], end="")


if __name__ == "__main__":
    img_origin = cv.imread("Experiment5-6/6-1.png")
    # 调整图片尺寸
    img = cv.resize(img_origin, None, fx=1 / 3, fy=1 / 3, interpolation=cv.INTER_CUBIC)

    # 1、对原始图片进行平滑和滤波处理
    # 高斯平滑
    img_gaussian = cv.GaussianBlur(img, (5, 5), 1)
    # 中值滤波
    img_median = cv.medianBlur(img_gaussian, 3)
    # 2、车牌定位
    # 2.1、
    region = location(img_median)
    # print(region)
    # 在原始图像中用红色方框标注
    img_showRect = img.copy()
    img_showRect = cv.drawContours(img_showRect, [region], -1, (0, 0, 255), 3)
    display("img_showRect", img_showRect)

    # 将车牌区域提取出来
    region_real = region * 3
    license_region = img_origin[
        np.min(region_real[:, 1]) : np.max(region_real[:, 1]) + 5,
        np.min(region_real[:, 0]) : np.max(region_real[:, 0]) + 10,
        :,
    ]
    display("license_region", license_region)

""" 
    # 2.2、对车牌区域进行角点校正
    # 原始车牌的四个角点(左下、左上、右下、右上，先列后行)
    pts1 = np.float32([[7, 47], [9, 23], [100, 35], [102, 9]])
    # 变换后分别在左下、左上、右下、右上四个点
    pts2 = np.float32([[0, 50], [0, 0], [110, 50], [110, 0]])  # 对应resize后的图像尺寸大小
    # 生成透视变换矩阵
    M = cv.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst = cv.warpPerspective(license_region, M, (110, 50))
    plt.subplot(121), plt.imshow(license_region[:, :, ::-1]), plt.title("input")  # img[:, :, ::-1]是将BGR转化为RGB
    plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title("output")
    plt.show()
 """
    # 3、分割区域灰度化、二值化
    license_region_gray = cv.cvtColor(license_region, cv.COLOR_BGR2GRAY)
    # 直方图均衡化
    license_region_gray = hist_equ(license_region_gray)
    license_region_binary = cv.threshold(license_region_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    # display("license_region_binary", license_region_binary)

    # 4、车牌分割
    license_region_binary[0:12, :] = 0
    license_region_binary[license_region_binary.shape[0] - 10 : license_region_binary.shape[0], :] = 0
    license_region_binary[:, 0:20] = 0
    license_region_binary[:, license_region_binary.shape[1] - 18 : license_region_binary.shape[1]] = 0
    license_region_binary[:, 126:130] = 0
    display("license_region_binary", license_region_binary)
    char_region_col = segmentation(license_region_binary)

    # 5、车牌识别
    # 5.1、字符提取
    for row in range(0, 7):  # 左闭右开
        temp = char_region_col[row, :]
        index = 0
        for i in range(temp.shape[0]):  # 去除列索引中多余的0
            if temp[i] == 0:
                index = i
                break
        col_temp = temp[0:index]
        temp_img = license_region_binary[:, np.min(col_temp) : np.max(col_temp) + 1]
        t = np.nonzero(np.sum(temp_img, axis=1))
        if row == 0:
            license_province1 = temp_img[t, :]  # 汉字后续扩展成40*40
            license_province1 = license_province1[0, :, :]
            license_province1 = char_extraction(license_province1)
            license_province1 = np.uint8(license_province1)
        if row == 1:
            license_province2 = temp_img[t, :]  # 字母和数字后续扩展成40*40
            license_province2 = license_province2[0, :, :]
            license_province2 = char_extraction(license_province2)
            license_province2 = np.uint8(license_province2)
        if row == 2:
            license_number1 = temp_img[t, :]
            license_number1 = license_number1[0, :, :]
            license_number1 = char_extraction(license_number1)
            license_number1 = np.uint8(license_number1)
        if row == 3:
            license_number2 = temp_img[t, :]
            license_number2 = license_number2[0, :, :]
            license_number2 = char_extraction(license_number2)
            license_number2 = np.uint8(license_number2)
        if row == 4:
            license_number3 = temp_img[t, :]
            license_number3 = license_number3[0, :, :]
            license_number3 = char_extraction(license_number3)
            license_number3 = np.uint8(license_number3)
        if row == 5:
            license_number4 = temp_img[t, :]
            license_number4 = license_number4[0, :, :]
            license_number4 = char_extraction(license_number4)
            license_number4 = np.uint8(license_number4)
        if row == 6:
            license_number5 = temp_img[t, :]
            license_number5 = license_number5[0, :, :]
            license_number5 = char_extraction(license_number5)
            license_number5 = np.uint8(license_number5)

    cv.imshow("license_province1", license_province1)
    cv.imshow("license_province2", license_province2)
    cv.imshow("license_number1", license_number1)
    cv.imshow("license_number2", license_number2)
    cv.imshow("license_number3", license_number3)
    cv.imshow("license_number4", license_number4)
    cv.imshow("license_number5", license_number5)

    cv.waitKey(0)
    # cv.hconcat()
    cv.destroyAllWindows()

    # 5.2、生成模板
    # 读取所有的汉字并生成模板
    Chinese_character = open("Experiment5-6/汉字.txt", encoding="gbk").read()
    Chinese_character = Chinese_character.split("\n")
    Chinese_char_template = template_generation("Experiment5-6/汉字", len(Chinese_character))
    # 读取所有的数字并生成模板
    Number_character = open("Experiment5-6/数字.txt", encoding="gbk").read()
    Number_character = Number_character.split("\n")
    Number_char_template = template_generation("Experiment5-6/数字", len(Number_character))
    # 读取所有的字母并生成模板
    Alphabet_character = open("Experiment5-6/英文.txt", encoding="gbk").read()
    Alphabet_character = Alphabet_character.split("\n")
    Alphabet_char_template = template_generation("Experiment5-6/英文", len(Alphabet_character))

    # 5.3、字符识别
    char_recognition()
