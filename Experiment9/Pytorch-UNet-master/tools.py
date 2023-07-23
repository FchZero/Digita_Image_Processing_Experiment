import os
import cv2 as cv

src_path = 'Experiment9\\dataset\\train\\src'
new_src_path = 'Experiment9\\dataset\\train\\new_src'
label_path = 'Experiment9\\dataset\\train\\labels'
new_label_path = 'Experiment9\\dataset\\train\\new_labels'
i=0
for root, dirs, files in os.walk(src_path):
    for file in files:
        src_file_path = os.path.join(src_path, file)
        label_file = file.replace('sat.jpg', '')
        label_file_path = os.path.join(src_path, file)
        label = cv.imread(label_file_path, -1)
        ret, label = cv.threshold(label, 127, 255, cv.THRESH_BINARY)
        label = label / 255
        new_name = file
        new_label_file_path = os.path.join(new_label_path, new_name)
        cv.imwrite(new_label_file_path, label)
        print(i)
        i = i + 1
