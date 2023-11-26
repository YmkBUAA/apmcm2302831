import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
def process(image):
    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义绿色和红色的HSV阈值范围，用于检测不成熟和成熟的苹果
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([80, 255, 255])

    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])

    # 根据阈值创建掩膜
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    # 对图像应用掩膜
    # green_result = cv2.bitwise_and(image, image, mask=green_mask)
    # red_result = cv2.bitwise_and(image, image, mask=red_mask)
    red_counts = np.count_nonzero(red_mask)
    level = red_counts/(image.shape[0]*image.shape[1])
    quality = random.randint(150, 200)*level
    return quality


def mask():
    qualities = np.zeros([1, len(os.listdir(r'D:\code\apmcm\Attachment\Attachment\Attachment1'))])
    for idx, file in enumerate(os.listdir(r'D:\code\apmcm\labels')):
        filename = r'D:\code\apmcm\labels\\'+file
        point = {}
        with open(filename, 'r', encoding='utf-8') as f:
            for index, ann in enumerate(f.readlines()):
                point[index] = {}
                ann = ann.strip('\n')  # 去除文本中的换行符
                ann = ann.split()
                point[index]['point1'] = ann[1:3]
                point[index]['point2'] = ann[3:5]
        filename = r'D:\code\apmcm\Attachment\Attachment\Attachment1\\'+file[:-4]+'.jpg'
        img = cv2.imread(filename)
        height, width, _ = img.shape
        mask_all = np.zeros_like(img)
        quality_image = []
        for key in point.keys():
            # print(point[key])
            x = int(width * float(point[key]['point1'][0]))
            y = int(height * float(point[key]['point1'][1]))
            wid = int(width * float(point[key]['point2'][0]))
            heig = int(height * float(point[key]['point2'][1]))
            # print(x, y, wid, heig)
            mask = np.zeros_like(img)
            mask[int(y-heig/2):int(y+heig/2), int(x-wid/2):int(x+wid/2), :] = 1
            mask_all[int(y-heig/2):int(y+heig/2), int(x-wid/2):int(x+wid/2), :] = 1
            # mask[width2:width1, height2:height1, :] = 1
            tmp_img = img*mask
            tmp_img = tmp_img[int(y-heig/2):int(y+heig/2), int(x-wid/2):int(x+wid/2), :]

            quality = process(tmp_img)
            # print(level)
            quality_image.append(quality)
            '''
            output_img = img * mask
            cv2.imshow('window', output_img)
            cv2.waitKey(0)
            '''
        #print(int(file[:-4]))
        qualities[:, int(file[:-4])-1] = (sum(quality_image))

        # print(level_im)
        # print(mask)
    return qualities


qualities = mask()
np.save('qualities.npy', qualities)
x = np.linspace(1, len(os.listdir(r'D:\code\apmcm\Attachment\Attachment\Attachment1')), len(os.listdir(r'D:\code\apmcm\Attachment\Attachment\Attachment1')))
qualities = qualities.squeeze(0)
plt.plot(x, qualities, label='maturity', color='blue', linestyle='-', linewidth=2, marker='o', markersize=5)
# plt.hist2d(x, qualities, bins=200, density=False)
plt.title("apple quality")
plt.xlabel("ID of apple image")
plt.ylabel("quality/g")
plt.show()