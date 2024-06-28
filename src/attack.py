import math
import asyncio
import torch
from torch import nn
import random
from torch import optim
import numpy
from PIL import Image
from torchvision import transforms
import os
from yolov3_helper import Helper as YoLov3Helper
import cv2
from tools.tools import *
from yolo import YOLO
from constant import *
import argparse


def create_astroid_mask(yolo, net, image_path, box_scale, shape=(416, 416)):
    mask = torch.zeros(*shape, 3)

    img = Image.open(img_path).convert('RGB')

    resize_small = transforms.Compose([
            transforms.Resize((416, 416)),
    ])

    img1 = resize_small(img)
    h, w = numpy.array(img).shape[:2]

    boxes = yolo.detect_box(img=img1)

    h, w = numpy.array(img).shape[:2]

    grids = boxes
    # print(mask.size()) [416*416*3]
    mask = torch.zeros(*shape, 3)
    visited_mask = torch.zeros(*shape, 3)
    num = 0
    for _, (y1, x1, y2, x2) in enumerate(grids):

        if num>9:
            break
        x1 = int(np.clip(x1, 0, 415))
        x2 = int(np.clip(x2, 0, 415))

        y1 = int(np.clip(y1, 0, 415))
        y2 = int(np.clip(y2, 0, 415))

        print("x1, y1, x2, y2", x1, y1, x2, y2)
        y_middle = (y1+y2)//2
        x_middle = (x1+x2)//2

        # shrink box
        box_h, box_w = int((y2-y1)*box_scale), int((x2-x1)*box_scale)
        y11 = y_middle-box_h//2
        y22 = y_middle+box_h//2
        x11 = x_middle-box_w//2
        x22 = x_middle+box_w//2

        cross_line_x_len = x_middle-x11
        cross_line_y_len = y_middle-y11
        cross_line_len = max(y_middle-y11, x_middle-x11)
        y_step, x_step = cross_line_y_len/cross_line_len, cross_line_x_len/cross_line_len

        tmp_mask = torch.zeros(mask.shape)
        # print(tmp_mask.shape) [416*416*3]

        tmp_mask[y_middle, x11:x22, :] = 1
        tmp_mask[y11:y22, x_middle, :] = 1

        for i in range(1, cross_line_len):
            # 第三通道全部置为1
            tmp_mask[y_middle-int(i*y_step), x_middle-int(i*x_step), :] = 1
            tmp_mask[y_middle+int(i*y_step), x_middle-int(i*x_step), :] = 1
            tmp_mask[y_middle-int(i*y_step), x_middle+int(i*x_step), :] = 1
            tmp_mask[y_middle+int(i*y_step), x_middle+int(i*x_step), :] = 1

            tmp_mask[y_middle-int(i*0.5*y_step), x_middle-int(i*x_step), :] = 1
            tmp_mask[y_middle+int(i*0.5*y_step), x_middle-int(i*x_step), :] = 1
            tmp_mask[y_middle-int(i*y_step), x_middle+int(i*0.5*x_step), :] = 1
            tmp_mask[y_middle+int(i*y_step), x_middle+int(i*0.5*x_step), :] = 1

            tmp_mask[y_middle-int(i*y_step), x_middle-int(i*0.5*x_step), :] = 1
            tmp_mask[y_middle+int(i*y_step), x_middle-int(i*0.5*x_step), :] = 1
            tmp_mask[y_middle-int(i*0.5*y_step), x_middle+int(i*x_step), :] = 1
            tmp_mask[y_middle+int(i*0.5*y_step), x_middle+int(i*x_step), :] = 1

        before_area = tmp_mask.sum()
        after_area = (tmp_mask*(1-visited_mask)).sum()

        if float(after_area) / float(before_area) < 0.5:
            continue

        if (mask + tmp_mask).sum() > 3000 * 3: 
            break

        num += 1
        mask = mask + tmp_mask
        visited_mask[y1:y2, x1:x2, :] = 1

    print("mask sum", mask.sum())
    return mask

'''
创建全是横杠的patch
'''
def create_plan_mask(yolo, net, img_path, box_scale, shape=(416, 416)):
    # 初始化一个空的遮罩
    mask = torch.zeros(*shape, 3)

    # 打开并转换图像
    img = Image.open(img_path).convert('RGB') 

    # 调整图像大小为416x416
    resize_small = transforms.Compose([
        transforms.Resize((416, 416)),
    ])

    img1 = resize_small(img)

    # 检测边框
    boxes = yolo.detect_box(img=img1)

    grids = boxes

    (y1, x1, y2, x2) =  boxes[0]
    
    # 确保坐标在图像范围内
    x1 = int(np.clip(x1, 0, 415))
    x2 = int(np.clip(x2, 0, 415))
    y1 = int(np.clip(y1, 0, 415))
    y2 = int(np.clip(y2, 0, 415))

    # 计算边框的高度和宽度
    box_h, box_w = y2 - y1, x2 - x1

    # 三千个像素点全部用掉,但是需要保证横纵的大小都大于0
    pixel_num = 3000

    # 为了避免浮点误差需要向上取整
    interval_h = int(box_w * box_h / 3000) + 1
    if interval_h <= 1:
        interval_h = 2

    pixel_changed = 0
    # 在遮罩上添加像素点
    for i in range(int(box_h / interval_h)):
        for j in range(box_w):
            y = int(y1 + i * interval_h)
            x = int(x1 + j)
            if y < shape[0] and x < shape[1]:
                mask[y, x, :] = 1
                pixel_changed = pixel_changed + 1
                if(pixel_changed >= 3000):
                    break

        if(pixel_changed >= 3000):
            break            

    # 将结果输出到log.txt文件
    print(pixel_changed)
    with open('log.txt', 'a') as log_file:
        log_file.write(f'File: {img_path}, Pixels Changed: {pixel_changed}\n')

    return mask

'''
创建均匀分布的mask，只考虑第一个box.具体方式为：
从mask的左上角[0,0,:]开始，行方向每间隔h/(根号下3000)，列方向每间隔w/(3000)个像素点使得mask=1.

简化一下，只考虑一个box的情况。
box不够大小本来就没有3000怎么办? 使用box像素点的四分之一。

box的大小够3000呢?
就用3000。

综合考虑其实就是全都用3000，只要保证间隔都大于1即可。

'''
# def create_plan_mask(yolo, net, img_path, box_scale, shape=(416, 416)):
#     # 初始化一个空的遮罩
#     mask = torch.zeros(*shape, 3)

#     # 打开并转换图像
#     img = Image.open(img_path).convert('RGB') 

#     # 调整图像大小为416x416
#     resize_small = transforms.Compose([
#         transforms.Resize((416, 416)),
#     ])

#     img1 = resize_small(img)

#     # 检测边框
#     boxes = yolo.detect_box(img=img1)

#     grids = boxes

#     (y1, x1, y2, x2) =  boxes[0]
    
#     # 确保坐标在图像范围内
#     x1 = int(np.clip(x1, 0, 415))
#     x2 = int(np.clip(x2, 0, 415))
#     y1 = int(np.clip(y1, 0, 415))
#     y2 = int(np.clip(y2, 0, 415))

#     # 计算边框的高度和宽度
#     box_h, box_w = y2 - y1, x2 - x1

#     # 三千个像素点全部用掉,但是需要保证横纵的大小都大于0
#     pixel_num = 3000

#     # 为了避免浮点误差需要向上取整
#     interval_h = int(box_h / np.sqrt(pixel_num)) + 1
#     interval_w = int(box_w / np.sqrt(pixel_num)) + 1
#     interval_h = 2 if interval_h <= 2 else interval_h
#     interval_w = 2 if interval_w <= 2 else interval_w

#     pixel_changed = 0
#     # 在遮罩上添加像素点
#     for i in range(int(box_h / interval_h)):
#         for j in range(int(box_w / interval_w)):
#             y = int(y1 + i * interval_h)
#             x = int(x1 + j * interval_w)
#             if y < shape[0] and x < shape[1]:
#                 mask[y, x, :] = 1
#                 pixel_changed = pixel_changed + 1
#                 if(pixel_changed >= 3000):
#                     break

#         if(pixel_changed >= 3000):
#             break            

#     # 将结果输出到log.txt文件
#     print(pixel_changed)
#     with open('log.txt', 'a') as log_file:
#         log_file.write(f'File: {img_path}, Pixels Changed: {pixel_changed}\n')

#     return mask

def get_delta(w):
    w = torch.clamp(w, 0, 255)
    return w

def specific_attack(yolov3_helper, img_path, mask, save_image_dir):
    img = cv2.imread(img_path)
    img = torch.from_numpy(img).float()

    t, max_iterations = 0, 600
    eps = 1
    w = torch.zeros(img.shape).float() + 127
    w.requires_grad = True
    success_attack = False
    min_object_num = 1000
    min_img = img

    while t < max_iterations:
        t += 1

        patch_connecticity = torch.abs(get_delta(w) - img).sum(-1) == 0
        patch = get_delta(w)
        patch[patch_connecticity] += 1

        patch_img = img * (1 - mask) + patch * mask
        patch_img = patch_img.to(device)

        attack_loss, object_nums = yolov3_helper.attack_loss(patch_img)

        if min_object_num > object_nums:
            min_object_num = object_nums
            min_img = patch_img

        if object_nums == 0:
            success_attack = True
            break

        if t % 20 == 0:
            print("t: {}, attack_loss:{}, object_nums:{}".format(t, attack_loss, object_nums))

        attack_loss.backward()

        w = w - eps * w.grad.sign()
        w = w.detach()
        w.requires_grad = True

    min_img = min_img.detach().cpu().numpy()

    with open('log_t.txt', 'a') as log_file:
        log_file.write(f'File: {img_path}, t: {t}\n')

    # 保存图像，攻击失败时在文件名中添加 "_fail"
    save_path = os.path.join(save_image_dir, img_path.split("/")[-1])
    if not success_attack:
        save_path = save_path.replace(".", "_fail.")

    cv2.imwrite(save_path, min_img)

    return success_attack


if __name__ == "__main__":
    random.seed(30)

    yolo = YOLO()

    box_scale = 1.0

    yolov3_helper = YoLov3Helper()
    model_helpers = [yolov3_helper]
    success_count = 0

    save_image_dir = "adv_images_v5"

    os.system("mkdir -p {}".format(save_image_dir))


    for i, img_path in enumerate(os.listdir("images")):
        img_path_ps = os.listdir(save_image_dir)
        if img_path in img_path_ps:
            success_count += 1
            continue
        if img_path.replace(".", "_fail.") in img_path_ps: 
            continue

        print("img_path", img_path)
            
        img_path = os.path.join("images", img_path)
        mask = create_plan_mask(yolo, yolov3_helper.net, img_path, box_scale)

        success_attack = specific_attack(yolov3_helper, img_path, mask, save_image_dir)

        if success_attack: 
            success_count += 1

        print("success: {}/{}".format(success_count, i + 1))