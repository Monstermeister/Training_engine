import tensorflow as tf

import cv2

import random
import math


import matplotlib.pyplot as plt

from pycocotools.coco import COCO
import os
import numpy as np

from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Dataset(object):
    def __init__(self, data_dir=None, json_file=None, name=None, input_size=(416, 416), batch_size=1):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
          data_dir (str): dataset root directory
          json_file (str): COCO json file name
          name (str): COCO data name (e.g. 'train2017' or 'val2017')
          input_size (int): target image size after pre-processing
          preproc: data augmentation strategy
        # """
        self.data_dir = data_dir
        self.json_file = json_file
        self.batch_size = batch_size
        # self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        # self.coco = COCO(os.path.join(self.json_file, "instances_", name, ".json"))
        self.coco = COCO(self.json_file)

        self.ids = self.load_ids()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.input_size = input_size
        self.annotations = self.load_coco_annotation()

    # Result: Array of (res, img_info, resized_info, file_name)
    #   res: Array of (bbox[0:4], class)
    #   file_name: image file name
    def load_coco_annotation(self):
        """
        Fill in annotations from image IDs
        """
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_ids(self):
        """
        Get image Ids which have 3:4 aspect ratio

        """
        image_ids = []
        # image_ids = [9, 25, 30, 34]

        for img_id in self.coco.getImgIds():
            image_ids.append(img_id)

        return image_ids

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.input_size[0] / height, self.input_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    ## Load image
    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def read_img(self, index):
        return self.load_resized_img(index)

    def get_dataset(self):
        img_list = []
        anno_list = []
        for idx, annotation in enumerate(self.annotations):
            box_info = annotation[0]
            img_list.append(self.read_img(idx))
            anno_list.append(box_info)
            print(idx)
        # res.append([self.read_img(idx), box_info)]
        #print('index:',idx,'image shape:',img_list[idx].shape,'anno_shape:',anno_list[idx].shape)
        return [img_list, anno_list]


###############################
class Dataloader(object):

    def __init__(self, img_list, anno_list, batch_size,
                 mosaic_prob=0.5, mosaic_scale=(0.5, 1.5),
                 rot_degree=10.0, translate=0.1, shear=2.0,
                 flip=0.5, HSV=1.0,
                 input_size=(416, 416)):

        self.img_list = img_list
        self.anno_list = anno_list
        self.mosaic_prob = mosaic_prob
        self.rot_degree = rot_degree
        self.translate = translate
        self.flip = flip
        self.shear = shear
        self.HSV = HSV
        self.input_size = input_size
        self.batch_size = batch_size
        self.mosaic_scale = mosaic_scale

    def _augmentation(self):

        inp_list = deepcopy(self.img_list)  ## Deepcopy not working self.img_list.deepcopy()
        tar_list = deepcopy(self.anno_list)  ## Deepcopy not working self. anno_list.deepcopy()
        data_num = len(inp_list)
        input_h, input_w = self.input_size[0], self.input_size[1]
        inputs = []
        targets = []
        for idx, (img, bbox) in enumerate(zip(inp_list, tar_list)):

            inp = img
            tar = bbox

            if random.random() < self.mosaic_prob:
                mosaic_labels = []

                y_c = int(random.uniform(0.5 * input_h, 1.5 * input_h))
                x_c = int(random.uniform(0.5 * input_w, 1.5 * input_w))

                mosaic_idxs = [idx] + [random.randint(0, data_num - 1) for _ in range(3)]

                for mosaic_idx, index in enumerate(mosaic_idxs):
                    m_image, m_labels = inp_list[index], tar_list[index]
                    h0, w0 = m_image.shape[:2]
                    scale = min(1. * input_h / h0, 1. * input_w / w0)
                    m_image = cv2.resize(
                        m_image, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)

                    h, w, c = m_image.shape[:3]

                    ############################ mosaic image
                    #### top left
                    if mosaic_idx == 0:
                        mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                        x1, y1, x2, y2 = max(x_c - w, 0), max(y_c - h, 0), x_c, y_c
                        s_x1, s_y1, s_x2, s_y2 = w - (x2 - x1), h - (y2 - y1), w, h
                    if mosaic_idx == 1:
                        x1, y1, x2, y2 = x_c, max(y_c - h, 0), min(x_c + w, input_w * 2), y_c
                        s_x1, s_y1, s_x2, s_y2 = 0, h - (y2 - y1), min(w, x2 - x1), h
                    if mosaic_idx == 2:
                        x1, y1, x2, y2 = max(x_c - w, 0), y_c, x_c, min(input_h * 2, y_c + h)
                        s_x1, s_y1, s_x2, s_y2 = w - (x2 - x1), 0, w, min(y2 - y1, h)
                    if mosaic_idx == 3:
                        x1, y1, x2, y2 = x_c, y_c, min(x_c + w, input_w * 2), min(y_c + h, input_h * 2)
                        s_x1, s_y1, s_x2, s_y2 = 0, 0, min(w, x2 - x1), min(h, y2 - y1)

                    mosaic_img[y1:y2, x1:x2] = m_image[s_y1:s_y2, s_x1:s_x2]

                    pos_w, pos_h = x1 - s_x1, y1 - s_y1

                    if m_labels.size > 0:
                        m_labels[:, 0] = scale * m_labels[:, 0] + pos_w
                        m_labels[:, 1] = scale * m_labels[:, 1] + pos_h
                        m_labels[:, 2] = scale * m_labels[:, 2] + pos_w
                        m_labels[:, 3] = scale * m_labels[:, 3] + pos_h
                    mosaic_labels.append(m_labels)

                if len(mosaic_labels):
                    mosaic_labels = np.concatenate(mosaic_labels, 0)
                    np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                    np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                    np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                    np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
                ################################
                inp = mosaic_img
                tar = mosaic_labels

                ############### rotation:10, translate 0.1, scale=(0.5,1.5), shear 2.0
                angle = random.uniform(-self.rot_degree, self.rot_degree)
                scale = random.uniform(self.mosaic_scale[0], self.mosaic_scale[1])

                R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

                M = np.ones([2, 3])

                shear_x = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
                shear_y = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 100)

                M[0] = R[0] + shear_y * R[1]
                M[1] = R[1] + shear_x * R[0]

                translation_x = random.uniform(-self.translate, self.translate) * input_w
                translation_y = random.uniform(-self.translate, self.translate) * input_h

                M[0, 2] = translation_x
                M[1, 2] = translation_y
                inp = cv2.warpAffine(inp, M, dsize=self.input_size, borderValue=(114, 114, 114))  ######################
                ##########################
                if len(tar) > 0:
                    corner_points = np.ones((4 * len(tar), 3))
                    corner_points[:, :2] = tar[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(4 * len(tar), 2)
                    corner_points = corner_points @ M.T
                    corner_points = corner_points.reshape(len(tar), 8)

                    corner_xs = corner_points[:, 0::2]
                    corner_ys = corner_points[:, 1::2]
                    new_bboxes = (
                        np.concatenate(
                            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
                        )
                        .reshape(4, len(tar))
                        .T
                    )
                    #####
                    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, input_w)
                    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, input_h)
                    #####
                    tar[:, :4] = new_bboxes  #####################

                ##### flip HSV
                boxes = tar[:, :4].copy()
                labels = tar[:, 4].copy()
                if len(boxes) == 0:
                    padded_labels = np.zeros((120, 5), dtype=np.float32)
                    inp_t, ratio_o = self._padding(inp, self.input_size)
                else:
                    ####### copy for empty boxes
                    inp_o = inp.copy()
                    tar_o = tar.copy()
                    height_o, width_o, _ = inp_o.shape
                    boxes_o = tar_o[:, :4]
                    labels_o = tar_o[:, 4]
                    # bbox_o; xyxy -> c_x,c_y,w,h
                    boxes_o = self._xyxy2cxcywh(boxes_o)

                    if random.random() < self.HSV:
                        self._hsv(inp)
                        ###
                    if random.random() < self.flip:
                        inp, boxes = self._mirror(inp, boxes)

                        #####
                    height, width, _ = inp.shape
                    inp_t, ratio = self._padding(inp, self.input_size)

                    boxes = self._xyxy2cxcywh(boxes)
                    boxes *= ratio

                    mask = np.minimum(boxes[:, 2], boxes[:, 3]) > 1

                    boxes_t = boxes[mask]
                    labels_t = labels[mask]
                    ###### boxes empty for objectness
                    if len(boxes_t) == 0:
                        inp_t, ratio_o = self._padding(inp_o, self.input_size)
                        boxes_o *= ratio_o
                        boxes_t = boxes_o
                        labels_t = labels_o
                    labels_t = np.expand_dims(labels_t, 1)

                    tar_t = np.hstack((boxes_t, labels_t))
                    padded_labels = np.zeros((120, 5))

                    padded_labels[range(len(tar_t))[:120]] = tar_t[:120]
                    padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)

            ######################### flip, hsv
            else:
                boxes = tar[:, :4].copy()
                labels = tar[:, 4].copy()
                if len(boxes) == 0:
                    padded_labels = np.zeros((120, 5), dtype=np.float32)
                    inp_t, ratio_o = self._padding(inp, self.input_size)

                else:
                    inp_o = inp.copy()
                    tar_o = tar.copy()
                    height_o, width_o, _ = inp_o.shape
                    boxes_o = tar_o[:, :4]
                    labels_o = tar_o[:, 4]
                    # bbox_o; xyxy -> c_x,c_y,w,h
                    boxes_o = self._xyxy2cxcywh(boxes_o)
                    if random.random() < self.HSV:
                        self._hsv(inp)
                    if random.random() < self.flip:
                        inp, boxes = self._mirror(inp, boxes)
                        #######
                    height, width, _ = inp.shape
                    inp_t, ratio = self._padding(inp, self.input_size)

                    boxes = self._xyxy2cxcywh(boxes)
                    boxes *= ratio

                    mask = np.minimum(boxes[:, 2], boxes[:, 3]) > 1

                    boxes_t = boxes[mask]
                    labels_t = labels[mask]

                    if len(boxes_t) == 0:
                        inp_t, ratio_o = self._padding(inp_o, self.input_size)
                        boxes_o *= ratio_o
                        boxes_t = boxes_o
                        labels_t = labels_o

                    labels_t = np.expand_dims(labels_t, 1)

                    tar_t = np.hstack((boxes_t, labels_t))
                    padded_labels = np.zeros((120, 5))

                    padded_labels[range(len(tar_t))[:120]] = tar_t[:120]
                    padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)



            inputs.append(inp_t)
            targets.append(padded_labels)

        return inputs, targets

    def _padding(self, img, input_size):
        if len(img.shape) == 3:
            padding_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padding_img = np.ones(input_size, dtype=np.uint8) * 114
        r = min((input_size[0] / img.shape[0]), (input_size[1] / img.shape[1]))  ## (input_size[1],img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR, ).astype(np.uint8)
        padding_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img

        padding_img = np.ascontiguousarray(padding_img, dtype=np.float32)

        return padding_img, r

    def _xyxy2cxcywh(self, bboxes):
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
        return bboxes

    def _hsv(self, img, hgain=5, sgain=30, vgain=30):
        hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
        hsv_augs *= np.random.randint(0, 2, 3)
        hsv_augs = hsv_augs.astype(np.int16)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

    ###########
    def _mirror(self, image, boxes):
        _, width, _ = image.shape
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes

    def generator(self):

        img_list, anno_list = self._augmentation()
        print("debug_augmentation")
        #image_list=[]
        #annot_list =[]
        #for i,(img, anno)  in enumerate(zip(img_list,anno_list)):
        #    image_list.append(tf.convert_to_tensor(img))
        #    annot_list.append(tf.convert_to_tensor(anno))


        #with tf.device("CPU"):
        print("debug data slicing")
        dataset = tf.data.Dataset.from_tensor_slices((img_list, anno_list))
        print("debug2")



        iteration = (len(dataset) // self.batch_size) if (len(dataset) % self.batch_size) == 0 else (len(dataset) // self.batch_size) + 1



        return dataset.shuffle(len(dataset)).repeat().batch(self.batch_size).take(iteration)


def visualize(img_list, anno_list):
    count = 0
    for i, (img, anno) in enumerate(zip(img_list, anno_list)):
        count += 1
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB)
        for j in range(len(anno)):  # 120
            xltop = int(anno[j][0]) - int(anno[j][2] * 0.5)
            yltop = int(anno[j][1]) + int(anno[j][3] * 0.5)
            xrbot = int(anno[j][0]) + int(anno[j][2] * 0.5)
            yrbot = int(anno[j][1]) - int(anno[j][3] * 0.5)
            print(xltop, yltop, xrbot, yrbot)
            cv2.rectangle(img, (xltop, yltop), (xrbot, yrbot), color=(255, 0, 0), thickness=1)
            print(count)
        plt.imshow(img)
        plt.savefig("/home/youngsang/work/Training_engine/Image_Test_COCO_Keti/" + '_' + str(i) + ".png")
        # if count == 4:
        #     break;
    return img

os.environ["CUDA_VISIBLE_DEVICES"]="0"

dataset = Dataset(data_dir="/data/image/keti/validate", json_file="/data/image/keti/annotate/instances_validate.json",
                  name="val")

img_list, anno_list = dataset.get_dataset()

dataloader = Dataloader(img_list, anno_list, batch_size=64, mosaic_prob=1.0, mosaic_scale=(1.0, 1.0),
                 rot_degree=0.0, translate=0.0, shear=0.0,
                 flip=0.0, HSV=0.0,
                 input_size=(416, 416))

dataset = dataloader.generator()

count =0
#for step,(img,anno) in enumerate(dataset):
#    count+=1
#    print("step",step,"data",img,anno)


for step,(img, anno_list) in enumerate(dataset):
    print("step",step)
    print("/n img",img.shape)
    print("/n anno_list", anno_list.shape)

img_list, anno_list = dataloader._augmentation()
visualize(img_list, anno_list)

#count=0
# for i,(img, anno)  in enumerate(zip(img_list, anno_list)):
#     count +=1
#     img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB)
#     for j in range(len(anno)): # 120
#         xltop = int(anno[j][0]) - int(anno[j][2]*0.5)
#         yltop = int(anno[j][1]) + int(anno[j][3]*0.5)
#         xrbot = int(anno[j][0]) + int(anno[j][2]*0.5)
#         yrbot = int(anno[j][1]) - int(anno[j][3]*0.5)
#         print(xltop, yltop, xrbot, yrbot)
#         cv2.rectangle(img, (xltop, yltop), (xrbot, yrbot), color=(255,0,0), thickness=1)
#         print(count)
#     plt.imshow(img)
#     plt.savefig("/home/youngsang/work/Aimf_yolox_tensorflow/Images_bbox/image" + '_' + str(i) + ".png")
#     if count == 4:
#         break;


# print(len(img_list)) # 전체 이미지 개수 5000
# print(len(anno_list)) # 전체 레이블 개수 5000
# print(len(anno_list[0])) # 이미지당 bbox 갯수 120
# print(len(anno_list[0][0] ~ anno_list[0][119])) # bbox 파라미터 수 5


