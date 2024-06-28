import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image
from torchvision import transforms
from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox

device = "cuda:0"

'''
训练自己的数据集必看注释！
'''
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'checkpoint/model.pth',
        "classes_path"      : 'model_data/classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [416, 416],
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
        "map_out_path"      : 'data/temp_map_out'
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        # show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   建立yolov3模型，载入yolov3模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        print(image_shape)
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image, []

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        dt_class_list = []
        dt_scores_list = []
        dt_boxes_list = []
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]
            
            top, left, bottom, right = box
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            dt_class_list.append(predicted_class)
            dt_scores_list.append(score)
            dt_boxes_list.append([left, top, right, bottom])

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image, [dt_class_list, dt_scores_list, dt_boxes_list]
        
    def generate_pre_file(self, img_name, box_info):
        if not len(box_info):
            return
        pre_cls_list, scores_list, boxes_list = box_info
        if not os.path.exists(self.map_out_path):
            os.makedirs(self.map_out_path)
        if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
            os.makedirs(os.path.join(self.map_out_path, "detection-results"))
        image_id = img_name.split('.')[0] 
        f = open(os.path.join(self.map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        for i, pre_cls in enumerate(pre_cls_list):
            score = scores_list[i]
            left, top, right, bottom = boxes_list[i] 
            f.write("%s %s %s %s %s %s\n" % (pre_cls, score, str(left), str(top), str(right),str(bottom)))
        f.close()
    
    def get_map(self):
        from utils.utils_map import get_coco_map
        import shutil
        with open('data/attacked_images_file.txt') as f:
            val_lines   = f.readlines()
        if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
            os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
        for annotation_line in val_lines:
            line        = annotation_line.split()
            image_id    = os.path.basename(line[0]).split('.')[0]
            gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
            #------------------------------#
            #   获得真实框txt
            #------------------------------#
            with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                for box in gt_boxes:
                    left, top, right, bottom, obj = box
                    obj_name = self.class_names[obj]
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Calculate Map.")
        # temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
        print("Get map done.")
        with open('data/map_record.txt','a') as f:
            f.write('MAP is ' + str(temp_map) + '\n')
        f.close
        shutil.rmtree(self.map_out_path)

    # 检测图片，获得得分和物体数量，用于对抗攻击

    def resize(self, img:torch.tensor):
        # resize
        img_pil = img.round().byte().detach().cpu().numpy()
        img_pil = Image.fromarray(img_pil.astype('uint8'), 'RGB')
        resize_small = transforms.Compose([
                transforms.Resize((416, 416)),
            ])
        img_pil = resize_small(img_pil)
        img_pil = np.array(img_pil)
        img_pil = torch.from_numpy(img_pil).float().to(device)

        img = img.permute(2,0,1)
        img = nn.functional.interpolate(img.unsqueeze(0), size=(416, 416),mode="bilinear",  align_corners=False)
        img = img.squeeze(0).permute(1,2,0)
        img = img + (img_pil-img.detach())
        return img

    def _input_transform(self, img:torch.tensor):
        ## bgr2rgb
        new_img = torch.zeros(img.shape).to(device)
        new_img[:,:,0] = new_img[:,:,0]+img[:,:,2]
        new_img[:,:,1] = new_img[:,:,1]+img[:,:,1]
        new_img[:,:,2] = new_img[:,:,2]+img[:,:,0]
        img = new_img

        # reisze
        img = self.resize(img)

        img = img.permute(2,0,1).contiguous()
        #img = img.float().div(255.0)
        img = img.float()/255.0
        img = img.unsqueeze(0)
        return img

    def attack_loss(self, image):
        self.net = self.net.eval().to(device)
        image = self._input_transform(image).to(device)

        image_shape =[416, 416]

        outputs = self.net(image)
        outputs = self.bbox_util.decode_box(outputs)
        results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                    image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
        if results[0] is None: 
            return image, []

        top_label   = np.array(results[0][:, 6], dtype = 'int32')
        top_conf    = results[0][:, 4] * results[0][:, 5]
        top_boxes   = results[0][:, :4]
    
        top_conf_tensor = torch.tensor(top_conf, requires_grad=True).to(device)
        return top_conf_tensor, len(top_label)

    def detect_box(self, img):
        if isinstance(img, Image.Image):
            width = img.width
            height = img.height
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
            img = img.view(1, 3, height, width)
            img = img.float().div(255.0)
        elif type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
        elif type(img) == torch.Tensor and len(img.shape) == 4:
            img = img
        else:
            print("unknow image type")
            exit(-1)

        image_shape = [416, 416]
        img = img.to(device)
        img = torch.autograd.Variable(img)
        outputs = self.net(img)
        outputs = self.bbox_util.decode_box(outputs)
        #---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        #---------------------------------------------------------#
        results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                
        top_boxes  = results[0][:, :4]

        return top_boxes