import cv2
import numpy as np


def normalize_values_to_0_255(values):
    # 归一化值到0-255范围
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0] * len(values)
    normalized_values = ((np.array(values) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized_values


def cal_light_map():
    l_map = np.zeros(image.shape[:2])
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            if mask[(h, w)][-1]:
                l_map[(h, w)] = image[(h, w)][0]
    # l_map = np.zeros(image.shape[:2])
    # l_map = image[:,:,0]
    cv2.imwrite(f'process/{name}/light.png', l_map)


def cal_degree():
    image_degree = np.zeros(image.shape[:2], dtype=np.uint8)
    origin_degree = np.zeros(image.shape[:2], dtype=np.float32)
    cal_t = (lambda p, a, ab: np.dot(p - a, ab) / np.dot(ab, ab))
    stage_start_color = img_decay_level[0].astype(np.float32)
    stage_end_color = img_decay_level[1].astype(np.float32)
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            if mask[(h, w)][-1]:
                color = image[(h, w)]
                degree = np.clip(cal_t(color, stage_start_color, stage_end_color - stage_start_color), 0.0,
                                 0.999)
                origin_degree[(h, w)] = degree
                image_degree[(h, w)] = (degree * 255).astype(np.uint8)
            # else:
            # image_degree[(h, w)] = 0.0
    # values = list(dict.values())
    # normalized_values = normalize_values_to_0_255(values)
    # for (h, w), value in zip(dict.keys(), normalized_values):
    #     image_degree[h, w] = value
    #
    # image_degree = image_degree.astype(np.uint8)
    cv2.imwrite(f'process/{name}/degree.png', image_degree)

    return origin_degree.copy()



def cal_shading_map_distance():
    shading_map = np.zeros(image.shape[:2], dtype=np.uint8)
    stage_start_color = img_decay_level[0].astype(np.float32)
    stage_end_color = img_decay_level[1].astype(np.float32)
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            if mask[(h, w)][-1]:
                color = image[(h, w)]
                coord = (h, w)
                #先计算这个coord对应的期望颜色
                color_expectation = stage_start_color + (stage_end_color - stage_start_color) * degree_map[coord]
                # l_expectation = cal_L(color_expectation)
                # l_current = cal_L(color)
                l_expectation = color_expectation[0]
                l_current = color[0]
                distance = l_current - l_expectation
                # print(distance)
                shading_map[coord] = np.clip(distance + 127, 0, 255)
    cv2.imwrite(f'process/{name}/shading_map.png', shading_map)
    return shading_map


def cal_texture_variation_map():
    stage_start_color = img_decay_level[0][1:].astype(np.float32)
    stage_end_color = img_decay_level[1][1:].astype(np.float32)
    texture_variation_map = np.zeros(shape=(image.shape[0], image.shape[1], 2)).astype(np.uint8)
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            if mask[(h, w)][-1]:
                coord = (h, w)
                color = image[coord][1:]
                color_expectation = stage_start_color + (stage_end_color - stage_start_color) * degree_map[coord]
                variation = ((color - color_expectation) * (1 - degree_map[coord]))*1.2
                variation = (variation + 127).astype(np.uint8)
                texture_variation_map[coord] = variation

    l_channel = np.full((image.shape[0], image.shape[1]), 127, dtype=image.dtype)
    texture_variation_map = cv2.merge([l_channel, texture_variation_map])
    texture_variation_image = cv2.cvtColor(texture_variation_map, cv2.COLOR_LAB2BGR)

    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            if not mask[(h, w)][-1]:
                texture_variation_image[(h, w)] = (0, 0, 0)

    cv2.imwrite(f'process/{name}/texture_variation.png', texture_variation_image)
    return texture_variation_map


def merge_texture():
    texture_paths = [
        f'process/{name}/adjusted_region_3.png',
        # f'merge/{name}/adjusted_region_2.png',
        # f'merge/{name}/adjusted_region_3.png'
    ]
    variation_path = f'process/{name}/texture_variation.png'

    image_variation = cv2.imread(variation_path)
    image_variation = cv2.cvtColor(image_variation, cv2.COLOR_BGR2LAB)
    image_variation_ab = np.zeros(shape=(image_variation.shape[0], image_variation.shape[1], 2))
    image_variation_ab = image_variation[:, :, 1:].astype(np.int16) - 127

    for i, path in enumerate(texture_paths):
        image_texture = cv2.imread(path)
        image_texture = cv2.cvtColor(image_texture, cv2.COLOR_BGR2LAB)
        # image_texture = image_texture.astype(np.int16)
        # 将 A_image_lab 的后两个通道加到 B_image_lab 上
        A_a_channel = image_texture[:, :, 1]
        A_b_channel = image_texture[:, :, 2]
        B_a_channel = image_variation_ab[:, :, 0]
        B_b_channel = image_variation_ab[:, :, 1]

        # 将 a 和 b 通道的像素值相加
        result_a_channel = A_a_channel + B_a_channel
        result_b_channel = A_b_channel + B_b_channel

        # 确保结果不会溢出
        result_a_channel = np.clip(result_a_channel, 0, 255).astype(np.uint8)
        result_b_channel = np.clip(result_b_channel, 0, 255).astype(np.uint8)

        # 将结果重新组合成 LAB 图像
        image_texture = np.stack([image_texture[:, :, 0], result_a_channel, result_b_channel], axis=2)
        # 将结果图像从 LAB 转换回 BGR 格式
        image_texture = cv2.cvtColor(image_texture, cv2.COLOR_Lab2BGR)
        cv2.imwrite(f'process/{name}/output_merged_3.png', image_texture)


name = 'a1'

path_img = f'{name}/{name}.png'
path_mask = f'{name}/{name}.png'
image = cv2.imread(path_img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)

img_control = [(416,632),(444, 1030)]  #a1__


img_decay_level = []
print(image.shape[:2])
for pixel in img_control:
    img_decay_level.append(image[pixel])
# img_decay_level = [np.array([235, 129, 118]), np.array([144, 132, 121])]

# degree_map = cal_degree_only2D()
degree_map = cal_degree()
#
shading_map = cal_shading_map_distance()

cal_texture_variation_map()
merge_texture()
# cal_light_map()

