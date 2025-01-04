import time
import cv2 as cv
import numpy as np
from tqdm import tqdm
from numba import jit

global update_map


def update_degree_overall(degree):
    for h in range(degree.shape[0]):
        for w in range(degree.shape[1]):
            # 不考虑degree为0.0的区域，这些区域为背景
            if degree[h][w] != 0.0:
                # 整体更新
                degree[h][w] += 1.0 * 0.02
                if degree[h][w] > 1.0:
                    degree[h][w] = 1.0
    return degree

def layers_weathered_method():
    for r in tqdm(range(shape[0]), desc='layers_weathered'):
        for s in range(shape[1]):
            update_degree = update_map[r][s]
            # 为了保持原有纹理的样貌细节,当前的degree比最后一个阶段要大，那么就仍保持原样
            if degree_map[r][s] > region_degree[2]:
                final_l[r][s] = image_l[r][s]
                final_a[r][s] = image_a[r][s]
                final_b[r][s] = image_b[r][s]
            elif update_degree < region_degree[0]:
                alpha = (update_degree - temp_degree_map[r][s])/region_degree[2]
                beta = (update_degree-temp_degree_map[r][s])/(region_degree[0]-temp_degree_map[r][s])
                final_l[r][s] = ((1-beta)*image_l[r][s]+beta*output_region_1_l[r][s])*(alpha*shading_l[r][s]-alpha+1)
                final_a[r][s] = output_region_1_a[r][s] * beta + image_a[r][s] * (1 - beta)
                final_b[r][s] = output_region_1_b[r][s] * beta + image_b[r][s] * (1 - beta)
            elif region_degree[0] < update_degree < region_degree[1]:
                alpha = (update_degree - temp_degree_map[r][s]) / region_degree[2]
                beta = (update_degree - region_degree[0]) / (region_degree[1] - region_degree[0])
                final_l[r][s] = ((1-beta)*output_region_1_l[r][s]+beta*output_region_2_l[r][s])*(alpha*shading_l[r][s]-alpha+1)
                final_a[r][s] = output_region_2_a[r][s] * beta + output_region_1_a[r][s] * (1 - beta)
                final_b[r][s] = output_region_2_b[r][s] * beta + output_region_1_b[r][s] * (1 - beta)
            elif region_degree[1] < update_degree < region_degree[2]:
                alpha = (update_degree - temp_degree_map[r][s]) / region_degree[2]
                beta = (update_degree - region_degree[1]) / (region_degree[2] - region_degree[1])
                final_l[r][s] = ((1-beta)*output_region_2_l[r][s]+beta*output_region_3_l[r][s])*(alpha*shading_l[r][s]-alpha+1)
                final_a[r][s] = output_region_3_a[r][s] * beta + output_region_2_a[r][s] * (1 - beta)
                final_b[r][s] = output_region_3_b[r][s] * beta + output_region_2_b[r][s] * (1 - beta)
            elif update_degree > region_degree[2]:
                final_l[r][s] = output_region_3_l[r][s]*shading_l[r][s]
                final_a[r][s] = output_region_3_a[r][s]
                final_b[r][s] = output_region_3_b[r][s]

def blur(m_degree_map):
    propagation_map = cv.blur(m_degree_map, (200, 200))
    return propagation_map


@jit(nopython=True)
def temp_method(propagation_map, Ks, update_times, degree):
    # for _ in tqdm(range(update_times), desc='update_degree_map'):
    for _ in range(update_times):
        # 为了向四周扩张的更快，每次更新map之前重新进行一次均值滤波（相当于向四周扩张）
        for p in range(degree.shape[0]):
            for m in range(degree.shape[1]):
                # 不考虑degree为0.0的区域，这些区域为背景
                if degree[p][m] > 0.3:
                    temp_degree = degree[p][m] + propagation_map[p][m] * Ks
                    degree[p][m] = temp_degree if temp_degree <= 1.0 else 1.0
                elif degree[p][m] <= 0.3:
                    temp_degree = degree[p][m] + propagation_map[p][m] * Ks
                    degree[p][m] = temp_degree if temp_degree <= 1.0 else 1.0
    return degree


def update_degree(degree, update_times):
    # 更新degree map
    Ks = 0.30
    propagation_map = blur(degree)
    new_degree = temp_method(propagation_map, Ks, update_times, degree)
    return new_degree


def get_total_updated_degree_map():
    total_updated_map = np.ones((shape[0], shape[1]), dtype=float)
    return total_updated_map


if __name__ == '__main__':
    # 加载数据
    filename = 'a1'
    numbers = [1, 1, 1, 1, 1]
    # numbers = [33, 1, 1, 1, 1, 1, 1, 1, 1]
    length = len(numbers)
    flag = 0
    input_image = cv.imread(f'{filename}/{filename}.jpg')  # 原图
    shape = input_image.shape

    png_image = cv.imread(f'{filename}/{filename}.png', cv.IMREAD_UNCHANGED)
    # png_a = np.ones(png_image.shape[:2], dtype=png_image.dtype) * 255
    # 合并RGB和阿尔法通道
    # png_image = cv.merge((png_image, png_a))
    png_r, png_g, png_b, png_a = cv.split(png_image)

    # 区域风化度图
    region_degree = np.loadtxt(f'{filename}/region_degree.txt', dtype=float, delimiter=',')
    k = region_degree.shape[0] - 1
    print("k =", k)

    # 加载shading
    shading = cv.imread(f'{filename}/shading_map.png')
    shading_l, shading_a, shading_b = cv.split(shading)



    shading_l = shading_l.astype(np.float32)
    k1 = 160

    a = 127

    shading_l[shading_l < a] = 1 - (a - shading_l[shading_l < a]) / k1

    shading_l[shading_l >= a] = 1 + (shading_l[shading_l >= a] - a) / k1

    # print(shading_l[(978,196)])
    # degree
    # degree_map = np.loadtxt('../test_xu/' + filename + '/' + 'degree_map.txt', dtype=float, delimiter=',')  # degree
    degree_map = cv.imread(f'{filename}/degree_map.png', cv.IMREAD_GRAYSCALE) / 255.0



    #加载纹理
    output_region_1 = cv.imread(f'{filename}/output_region_1_.jpg')
    output_region_2 = cv.imread(f'{filename}/output_region_2_.jpg')
    output_region_3 = cv.imread(f'{filename}/output_merged_3.png')

    # 通道转换并分离
    output_region_1_lab = cv.cvtColor(output_region_1, cv.COLOR_BGR2Lab)
    output_region_1_l, output_region_1_a, output_region_1_b = cv.split(output_region_1_lab)
    if output_region_2 is not None:
        output_region_2_lab = cv.cvtColor(output_region_2, cv.COLOR_BGR2Lab)
        output_region_2_l, output_region_2_a, output_region_2_b = cv.split(output_region_2_lab)
    if output_region_3 is not None:
        output_region_3_lab = cv.cvtColor(output_region_3, cv.COLOR_BGR2Lab)
        output_region_3_l, output_region_3_a, output_region_3_b = cv.split(output_region_3_lab)


    # image通道分离
    lab_input_image = cv.cvtColor(input_image, cv.COLOR_BGR2Lab)
    image_l, image_a, image_b = cv.split(lab_input_image)
    degree_copy = degree_map.copy()

    # 归一化
    image_l = image_l / 255.0
    image_a = image_a / 255.0
    image_b = image_b / 255.0

    output_region_1_l = output_region_1_l / 255.0
    output_region_1_a = output_region_1_a / 255.0
    output_region_1_b = output_region_1_b / 255.0

    if output_region_2 is not None:
        output_region_2_l = output_region_2_l / 255.0
        output_region_2_a = output_region_2_a / 255.0
        output_region_2_b = output_region_2_b / 255.0

    if output_region_3 is not None:
        output_region_3_l = output_region_3_l / 255.0
        output_region_3_a = output_region_3_a / 255.0
        output_region_3_b = output_region_3_b / 255.0


    # save data
    final_l = np.zeros([image_l.shape[0], image_l.shape[1]])
    final_a = np.zeros([image_a.shape[0], image_a.shape[1]])
    final_b = np.zeros([image_b.shape[0], image_b.shape[1]])


    index = 1
    temp_degree_map = degree_map.copy()  # 复制一份
    update_map = degree_map.copy()

    for number in numbers:
        times = time.time()
        # update degree map
        update_map = update_degree(update_map, number)
        update_map = update_degree_overall(degree_copy)

        layers_weathered_method()

        final_lab = cv.merge([final_l * 255.0, final_a * 255.0, final_b * 255.0])
        final_bgr = cv.cvtColor(final_lab.astype('uint8'), cv.COLOR_Lab2BGR)

        # 将前景填充上去
        for i in tqdm(range(final_bgr.shape[0]), desc='fill foreground'):
            for j in range(final_bgr.shape[1]):
                # 将前景和像素值仍为0.01的地方用原始的值来代替
                if png_a[i][j] == 0 or update_map[i][j] == 0.01:
                    final_bgr[i][j][0] = input_image[i][j][0]
                    final_bgr[i][j][1] = input_image[i][j][1]
                    final_bgr[i][j][2] = input_image[i][j][2]

        # plt.subplot(2, length + 1, index + 1)#-------------------------
        # plt.imshow(final_bgr[:, :, (2, 1, 0)])#-------------------------
        index += 1
        cv.imwrite(f'{filename}/output/{str(int(times))[6:]}k1={k1},a={a}.jpg',
                   final_bgr)  # -------------------------------------------


    # plt.show()#-------------------------
