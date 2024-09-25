import random
import math
import cv2
import numpy as np
import os
from copy import deepcopy
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, data_dir=None, json_file=None, input_size=(416, 416), batch_size=1):
        self.data_dir = data_dir
        self.json_file = json_file
        self.batch_size = batch_size
        self.input_size = input_size
        self.coco = COCO(self.json_file)
        self.ids = self.load_ids()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.annotations = self.load_coco_annotation()
        
    def __len__(self):
        return self.num_imgs
    
    def load_ids(self):
        return self.coco.getImgIds()

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
        file_name = im_ann.get("file_name", "{:012}".format(id_) + ".jpg")
        return (res, img_info, resized_info, file_name)

    def load_coco_annotation(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, file_name)
        img = cv2.imread(img_file)
        if img is None:
            raise FileNotFoundError(f"File named {img_file} not found")
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
        return img_list, anno_list, self.ids


class DataAugmentor:
    def __init__(self, input_size=(416, 416), **kwargs):
        self.input_size = input_size
        self.mosaic_scale = kwargs.get('mosaic_scale', (1.0, 1.0))
        self.mosaic_prob = kwargs.get('mosaic_prob', 1.0)
        self.rot_degree = kwargs.get('rot_degree', 0.0)
        self.translate = kwargs.get('translate', 0.0)
        self.shear = kwargs.get('shear', 0.0)
        self.flip = kwargs.get('flip', 0.0)
        self.HSV = kwargs.get('HSV', 0.0)

    def augment(self, images, annotations):
        augmented_images = []
        augmented_annotations = []
        for img, anno in zip(images, annotations):
            augmented_img, augmented_anno = self.apply_augmentation(img, anno)
            augmented_images.append(augmented_img)
            augmented_annotations.append(augmented_anno)
        return augmented_images, augmented_annotations

    def apply_augmentation(self, images, annotations):
        inp_list = deepcopy([images])
        tar_list = deepcopy([annotations])
        data_num = len(inp_list)
        input_h, input_w = self.input_size[0], self.input_size[1]
        inputs = []
        targets = []

        for idx, (img, bbox) in enumerate(zip(inp_list, tar_list)):
            inp = img
            tar = bbox

            if random.random() < self.mosaic_prob:
                inp, tar = self._mosaic_augmentation(inp_list, tar_list, input_h, input_w, idx, data_num)
                
            inp, tar = self._rotate_translate_scale(inp, tar)
            inp, tar = self._flip_HSV(inp, tar, input_h, input_w)

            inputs.append(inp)
            targets.append(tar)

        return inputs, targets

    def _mosaic_augmentation(self, inp_list, tar_list, input_h, input_w, idx, data_num):
        mosaic_labels = []
        y_c = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        x_c = int(random.uniform(0.5 * input_w, 1.5 * input_w))
        mosaic_idxs = [idx] + [random.randint(0, data_num - 1) for _ in range(3)]

        mosaic_img = np.full((input_h * 2, input_w * 2, inp_list[0].shape[2]), 114, dtype=np.uint8)

        x1, y1, x2, y2 = self._calculate_mosaic_coordinates(0, x_c, y_c, inp_list[0].shape[1], inp_list[0].shape[0], input_w, input_h)
        mosaic_img[y1:y2, x1:x2] = inp_list[idx][:y2-y1, :x2-x1]

        for mosaic_idx, index in enumerate(mosaic_idxs[1:], 1):
            m_image, m_labels = inp_list[index], tar_list[index]
            h0, w0 = m_image.shape[:2]
            scale = min(1. * input_h / h0, 1. * input_w / w0)
            m_image = cv2.resize(m_image, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)

            h, w, c = m_image.shape[:3]

            x1, y1, x2, y2 = self._calculate_mosaic_coordinates(mosaic_idx, x_c, y_c, w, h, input_w, input_h)

            mosaic_img[y1:y2, x1:x2] = m_image[:y2-y1, :x2-x1]

            if m_labels.size > 0:
                m_labels[:, 0:4:2] = scale * m_labels[:, 0:4:2] + (x1 if mosaic_idx % 2 == 0 else 0)
                m_labels[:, 1:4:2] = scale * m_labels[:, 1:4:2] + (y1 if mosaic_idx < 2 else 0)
            mosaic_labels.append(m_labels)

        if mosaic_labels:
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            np.clip(mosaic_labels[:, 0:4:2], 0, 2 * input_w, out=mosaic_labels[:, 0:4:2])
            np.clip(mosaic_labels[:, 1:4:2], 0, 2 * input_h, out=mosaic_labels[:, 1:4:2])

        return mosaic_img, mosaic_labels

    def _calculate_mosaic_coordinates(self, mosaic_idx, x_c, y_c, w, h, input_w, input_h):
        if mosaic_idx == 0:
            return max(x_c - w, 0), max(y_c - h, 0), x_c, y_c
        elif mosaic_idx == 1:
            return x_c, max(y_c - h, 0), min(x_c + w, input_w * 2), y_c
        elif mosaic_idx == 2:
            return max(x_c - w, 0), y_c, x_c, min(input_h * 2, y_c + h)
        else:
            return x_c, y_c, min(x_c + w, input_w * 2), min(y_c + h, input_h * 2)

    def _rotate_translate_scale(self, inp, tar):
        angle = random.uniform(-self.rot_degree, self.rot_degree)
        scale = random.uniform(self.mosaic_scale[0], self.mosaic_scale[1])
        M = cv2.getRotationMatrix2D((0, 0), angle=angle, scale=scale)

        shear = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        M[0] += shear * M[1]
        M[1] += shear * M[0]

        translation_x = random.uniform(-self.translate, self.translate) * self.input_size[1]
        translation_y = random.uniform(-self.translate, self.translate) * self.input_size[0]
        M[0, 2] += translation_x
        M[1, 2] += translation_y

        inp = cv2.warpAffine(inp, M, dsize=self.input_size, borderValue=(114, 114, 114))

        if len(tar) > 0:
            # Transform bounding boxes
            tar[:, :4] = self._transform_bboxes(tar[:, :4], M, self.input_size)

        return inp, tar


    def _transform_bboxes(self, bboxes, M, input_size):
        transformed_bboxes = np.zeros_like(bboxes)

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]], dtype=np.float32)
            transformed_corners = cv2.transform(np.array([corners]), M)[0]

            x_coords = transformed_corners[:, 0]
            y_coords = transformed_corners[:, 1]

            transformed_bboxes[i] = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]

        transformed_bboxes[:, 0::2] = np.clip(transformed_bboxes[:, 0::2], 0, input_size[1])
        transformed_bboxes[:, 1::2] = np.clip(transformed_bboxes[:, 1::2], 0, input_size[0])

        return transformed_bboxes


    def _flip_HSV(self, inp, tar, input_h, input_w):
        if random.random() < self.HSV:
            self._hsv(inp)

        if random.random() < self.flip:
            inp, tar[:, :4] = self._mirror(inp, tar[:, :4])

        inp, ratio = self._padding(inp, self.input_size)
        tar[:, :4] *= ratio

        mask = np.minimum(tar[:, 2], tar[:, 3]) > 1
        tar = tar[mask]

        if len(tar) == 0:
            tar = np.zeros((120, 5), dtype=np.float32)
        else:
            tar = np.pad(tar, ((0, max(0, 120 - len(tar))), (0, 0)), mode='constant', constant_values=0)

        return inp, tar
    
    def _padding(self, img, input_size):
        padding_img = np.ones(input_size + (3,), dtype=np.uint8) * 114
        r = min((input_size[0] / img.shape[0]), (input_size[1] / img.shape[1]))
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padding_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img
        padding_img = np.ascontiguousarray(padding_img, dtype=np.float32)
        return padding_img, r

    def _hsv(self, img, hgain=5, sgain=30, vgain=30):

        hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
        hsv_augs *= np.random.randint(0, 2, 3)
        hsv_augs = hsv_augs.astype(np.int16)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

    def _mirror(self, image, boxes):

        _, width, _ = image.shape
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        return image, boxes    

class Dataloader:
    def __init__(self, batch_size, dataset, augmentor):
        self.batch_size = batch_size
        self.dataset = dataset
        self.augmentor = augmentor

    def __len__(self):
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, idx):
        batch_images, batch_annotations, ids = self.dataset.get_dataset()
        batch_images, batch_annotations = self.augmentor.augment(batch_images, batch_annotations)
        return batch_images, batch_annotations, ids
    
    def visualize_and_save(self, images, annotations, ids):
        for i, (img, anno) in enumerate(zip(images, annotations)):
            img = img.reshape(416,416,3)
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.savefig("/home/youngsang/work/Training_engine/Mixed_Image/Original_Image/" + 'Image_' + str(ids[i])+ ".png")

            # Visualize and save annotated image

            for bbox in anno:
                for j in range(len(bbox)):
                    x1 = bbox[j][0]
                    y1 = bbox[j][1]
                    x2 = bbox[j][2]
                    y2 = bbox[j][3]
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    print(x1,y1,x2,y2)
            plt.imshow(img)
            plt.savefig("/home/youngsang/work/Training_engine/Mixed_Image/Image_Test_Final/" + 'Image_' + str(ids[i])+ ".png")
            print(f'Image {ids[i]} saved with annotations.')


# # Example usage:
def main():
    data_dir = "/data/image/keti/validate"
    json_file = "/data/image/keti/annotate/instances_validate.json"
    dataset = Dataset(data_dir, json_file)
    # ('mosaic_scale', (1.0, 1.5))
    # ('mosaic_prob', 0.5)
    # ('rot_degree', 10.0)
    # ('translate', 0.1)
    # ('shear', 2.0)
    # ('flip', 0.5)
    # ('HSV', 1.0)
    augmentor = DataAugmentor(mosaic_prob=0.5)
    dataloader = Dataloader(batch_size=1, dataset=dataset, augmentor=augmentor)
    for i in range(len(dataloader)):
        images, annotations, ids = dataloader[i]
        images = np.array(images)
        dataloader.visualize_and_save(images, annotations, ids)
        if i == 0:
            break;

if __name__ == "__main__":
	main()
