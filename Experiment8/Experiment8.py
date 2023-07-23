import cv2
import numpy as np
from moviepy.editor import VideoFileClip

"""
直线可以表示为𝑦 = 𝑘𝑥 + 𝑞，这个是以 x,y 为坐标轴，现在考虑以 k,q 为坐标轴的坐标，
由𝑦 = 𝑘𝑥 + 𝑞可知，k,q 的关系也是线性关系，也就是说固定一个 x,y 值，可以得 k,q 坐标系里的一条直线，
对于 x,y 坐标系的 A，B 两点，就可以在 k,q 坐标系获得两条直线，在 k,q 坐标系的交点处，两条直线具有相同的 k,q 值，
映射到  k,q 标系，指的就是穿过A，B 两点直线的斜率 k 和截距  q。
霍夫变换，就是将黑白图片中的每个白色像素映射到 k,q 坐标系（也可以是极坐标系，原理一样）中，
然后统计 q，k 坐标系中每个交点处直线数量，最后进行排序，选择前 n个 k,q 值形成 n 条直线。
"""


blur_ksize = 5
canny_lthreshold = 50
canny_hthreshold = 150
# 霍夫变换超参数
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20


# 提取 ROI 区域：用一个梯形框将车道线存在的区域提取出来，这样可以去除额外的很多噪声
def roi_mask(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        mask_color = (255,) * channel_count
    else:
        mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


# 霍夫变换，返回画好车道线的图片：根据每条线段的斜率分为左右直线，因为左侧跟右侧的直线，一个斜率为正，
# 一个斜率为负，然后再去除一些偏离斜率均值较远的直线，最后再将左右侧的直线分别拟合成一条直线即可。
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lanes(line_img, lines)
    return line_img


# 首先将霍夫变换获得的直线进行分类，然后分别再拟合出左右两条直线
def draw_lanes(img, lines, color=None, thickness=8):
    if color is None:
        color = [255, 0, 0]
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)
    if len(left_lines) <= 0 or len(right_lines) <= 0:
        return img
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]
    left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])
    right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])
    cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
    cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)


# 去掉偏离均值较远的直线，这里的值是指斜率
def clean_lines(lines, threshold):
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


# 将左右两侧的直线拟合，返回左右车道线的顶点
def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)
    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))
    return [(xmin, ymin), (xmax, ymax)]


# 总的处理函数，输入一张图片，返回一张标记车道线的图片
def process_an_image(img):
    roi_vtx = np.array([[(0, img.shape[0]), (460, 325), (520, 325), (img.shape[1], img.shape[0])]])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    roi_edges = roi_mask(edges, roi_vtx)
    line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    return res_img


mp4_path = "Experiment1-8\8.mp4"
out_path = "Experiment1-8\8_out.mp4"
clip = VideoFileClip(mp4_path)
out_clip = clip.fl_image(process_an_image)
out_clip.write_videofile(out_path, audio=False)
