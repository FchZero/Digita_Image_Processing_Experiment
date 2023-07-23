import os
import cv2 as cv
import numpy as np


# 车牌定位函数
def location(image):
    # 多维的切片是在中括号中用逗号运算符, 将不同维上的操作分开，分割开后每个维度上单独维护即可
    # image[1:2, 3:4, 2]即第1、2行的第3、4列的第3个通道的像素值
    R = image[:, :, 2]
    G = image[:, :, 1]
    B = image[:, :, 0]
    region = []  # 2维矩阵存放符合阈值条件的点的行列号
    for i in range(image.shape[0]):  # shape[0]为行数（高度），对每一行进行遍历
        for j in range(image.shape[1]):  # shape[1]为列数（宽度），对每一列进行遍历
            # 可将 RGB 图像转化为 HSV 图像进行阈值比较，HSV 图像可使用阈值：H：190~245, S：0.35~1, V：0.3~1
            # 车牌分割参考阈值（RGB 图像）：
            if B[i, j] > 0:
                if ((R[i, j] / B[i, j]) < 0.35 and (G[i, j] / B[i, j]) < 0.9 and B[i, j] > 90) or (
                    (R[i, j] / B[i, j]) < 0.9 and (G[i, j] / B[i, j]) < 0.35 and B[i, j] < 90
                ):
                    region.append([i, j])
    region = np.array(region)

    # 索引去重，因为车牌区域直接框选，只需知道车牌的左上角和右下角的坐标即可，不需知道每个点的坐标
    # 行索引去重
    row_index = np.unique(region[:, 0])
    # 行索引计数（不含重复行索引）
    row_index_number = np.zeros(row_index.shape, dtype=np.uint8)
    for i in range(region.shape[0]):
        for j in range(row_index.shape[0]):
            if region[i, 0] == row_index[j]:
                row_index_number[j] = row_index_number[j] + 1
    row_index_number = row_index_number > 10  # 将误判的点去除
    row_index = row_index[row_index_number]

    # 列索引去重
    col_index = np.unique(region[:, 1])
    # 列索引计数（不含重复列索引）
    col_index_number = np.zeros(col_index.shape, dtype=np.uint8)
    for i in range(region.shape[0]):
        for j in range(col_index.shape[0]):
            if region[i, 1] == col_index[j]:
                col_index_number[j] = col_index_number[j] + 1
    col_index_number = col_index_number > 10  # 将误判的点去除
    col_index = col_index[col_index_number]

    region = np.array([[np.min(row_index), np.max(row_index)], [np.min(col_index), np.max(col_index)]])
    cv.rectangle(
        image,
        # 注意坐标格式为(x, y)，与数组索引相反，因为行数代表y，列数代表x
        pt1=(region[1, 0], region[0, 0]),
        pt2=(region[1, 1], region[0, 1]),
        color=(0, 0, 255),
        thickness=2,
    )
    cv.imshow("region", image)  # 在原image上框选车牌区域
    license_region = img[region[0, 0] : region[0, 1], region[1, 0] : region[1, 1]]
    # cv.imshow("license_region", license_region)  # 单独显示车牌区域
    cv.waitKey(0)
    return region, license_region


# 车牌分割函数
def segmentation(image):
    temp_col_index = []  # 1维数组存储含有字符的列的索引
    for col in range(image.shape[1]):
        # 根据灰度值在y轴的投影对车牌二值图像进行分割,因为在没有字符的区域，y方向上像素灰度和为0，在有字符的区域为灰度和非0
        # 为了避免误差影响，存在大于等于2个255的列判定存在字符
        if np.sum(image[:, col]) >= 2 * 255:
            temp_col_index.append(col)
    temp_col_index = np.array(temp_col_index)

    flag_i = 0  # 第flag_i个字符
    flag = 0  # 第flag_i个字符的起始列
    # 2维数组，第flag_i行代表第flag_i个字符的列索引，每行的列数为30，不足30的补0
    char_region_col = np.uint8(np.zeros([7, 30]))
    for j in range(temp_col_index.shape[0] - 1):
        # 相邻两个列索引的差值大于等于2，判定这两个列索引之间没有字符
        # 此时开始前一个字符的列索引的写入工作
        if temp_col_index[j + 1] - temp_col_index[j] >= 2:
            # 第flag_i个字符的列索引为flag到j+1
            temp = temp_col_index[flag : j + 1]
            temp = np.append(temp, np.zeros(30 - temp.shape[0]))  # 补成30维的向量
            temp = np.uint8(temp.reshape(1, 30))
            # 第flag_i个字符的列索引存入第flag_i行
            char_region_col[flag_i, :] = temp
            flag = j + 1
            flag_i = flag_i + 1
    # 每次循环结束才写入前一个字符的列索引，最后一个字符的列索引的写入工作需在循环外完成
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

    # 若提取到的含有字符的列之间不是相邻的(可初步解决川的分割问题)
    if col_index.shape[0] <= 3 or row_index.shape[0] <= 3:
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
        char_image = np.uint8(char_image)
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
    img = cv.imread("Experiment5-6/5.jpg")
    img = cv.imread("Experiment5-6/6-1.png")
    # 1、车牌定位
    region, license_region = location(img)

    # 2、分割区域灰度化、二值化
    license_region_gray = cv.cvtColor(license_region, cv.COLOR_BGR2GRAY)
    # cv.imshow('license_region_gray', license_region_gray)
    # cv.waitKey(0)
    # cv2.THRESH_OTSU是一种自适应阈值化方法，它能够自动选择最佳阈值。
    # 它基于Otsu's方法，使用最小二乘法处理像素点，该方法通过分析图像的灰度直方图来确定最佳阈值，以使分割后的图像具有最小类间方差。
    # 当将cv.THRESH_BINARY和cv.THRESH_OTSU结合使用时，可以自动选择最佳阈值，并将图像进行二值化处理。
    license_region_binary = cv.threshold(license_region_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    # cv.imshow("license_region_binary", license_region_binary)
    # cv.waitKey(0)

    # 3、车牌分割
    char_region_col = segmentation(license_region_binary)

    # 4、车牌识别
    # 4.1、字符提取
    char_region_row = np.uint8(np.zeros([7, 30]))
    for row in range(char_region_row.shape[0]):
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

    # 4.2、生成模板
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

    # 4.3、字符识别
    char_recognition()
