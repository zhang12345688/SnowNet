from random import sample, shuffle
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, clearimage_lines, input_shape, num_classes, epoch_length, \
                 mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.clearimage_lines = clearimage_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio

        self.epoch_now = -1
        self.length = len(self.annotation_lines)

        self.bbox_attrs = 5 + num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            clearlines = sample(self.clearimage_lines, 3)
            lines.append(self.annotation_lines[index])
            clearlines.append(self.clearimage_lines[index])
            shuffle(lines)
            image, box, clearimg = self.get_random_data_with_Mosaic(lines, self.input_shape, clearlines)
            if self.mixup and self.rand() < self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_2, box_2, clearimg_2 = self.get_random_data(lines[0], self.input_shape, clearlines[0], random=self.train)
                image, box, clearimg = self.get_random_data_with_MixUp(image, box, clearimg, image_2, box_2, clearimg_2)
        else:
            image, box, clearimg = self.get_random_data(
                self.annotation_lines[index],
                self.clearimage_lines[index],
                self.input_shape,
                random=self.train
            )

        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        clearimg = np.transpose(preprocess_input(np.array(clearimg, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        return image, labels_out, clearimg

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, clearimage_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        clearline = clearimage_line.split()
        image = Image.open(line[0])
        image = cvtColor(image)
        clearimg = Image.open(clearline[0])
        clearimg = cvtColor(clearimg)
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            clearimg = clearimg.resize((nw, nh), Image.BICUBIC)
            new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
            new_clearimg.paste(clearimg, (dx, dy))
            clear_image_data = np.array(new_clearimg, np.float32)

            if len(box) > 0:
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box, clear_image_data

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        clearimg = clearimg.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
        new_clearimg.paste(clearimg, (dx, dy))
        clearimg = new_clearimg

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            clearimg = clearimg.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        clear_image_data = np.array(clearimg, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        hue1, sat1, val1 = cv2.split(cv2.cvtColor(clear_image_data, cv2.COLOR_RGB2HSV))
        dtype1 = clear_image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        x1 = np.arange(0, 256, dtype=r.dtype)
        lut_hue1 = ((x1 * r[0]) % 180).astype(dtype)
        lut_sat1 = np.clip(x1 * r[1], 0, 255).astype(dtype)
        lut_val1 = np.clip(x1 * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        clear_image_data = cv2.merge((cv2.LUT(hue1, lut_hue1), cv2.LUT(sat1, lut_sat1), cv2.LUT(val1, lut_val1)))
        clear_image_data = cv2.cvtColor(clear_image_data, cv2.COLOR_HSV2RGB)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box, clear_image_data

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, clearimage_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        clear_image_datas = []
        box_datas = []
        index = 0
        for line, clear_line in zip(annotation_line, clearimage_line):
            line_content = line.split()
            clear_line_content = str(clear_line).split()
            image = Image.open(line_content[0])
            image = cvtColor(image)
            clear_image = Image.open(clear_line_content[0])
            clear_image = cvtColor(clear_image)
            iw, ih = image.size
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            new_clear_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_clear_image.paste(clear_image, (dx, dy))
            clear_image_data = np.array(new_clear_image)

            index = index + 1
            box_data = []
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            clear_image_datas.append(clear_image_data)
            box_datas.append(box_data)

        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_clear_image = np.zeros([h, w, 3])
        new_clear_image[:cuty, :cutx, :] = clear_image_datas[0][:cuty, :cutx, :]
        new_clear_image[cuty:, :cutx, :] = clear_image_datas[1][cuty:, :cutx, :]
        new_clear_image[cuty:, cutx:, :] = clear_image_datas[2][cuty:, cutx:, :]
        new_clear_image[:cuty, cutx:, :] = clear_image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        new_clear_image = np.array(new_clear_image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        hue1, sat1, val1 = cv2.split(cv2.cvtColor(new_clear_image, cv2.COLOR_RGB2HSV))
        dtype1 = new_clear_image.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        x1 = np.arange(0, 256, dtype=r.dtype)
        lut_hue1 = ((x1 * r[0]) % 180).astype(dtype)
        lut_sat1 = np.clip(x1 * r[1], 0, 255).astype(dtype)
        lut_val1 = np.clip(x1 * r[2], 0, 255).astype(dtype)

        new_clear_image = cv2.merge((cv2.LUT(hue1, lut_hue1), cv2.LUT(sat1, lut_sat1), cv2.LUT(val1, lut_val1)))
        new_clear_image = cv2.cvtColor(new_clear_image, cv2.COLOR_HSV2RGB)

        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes, new_clear_image

    def get_random_data_with_MixUp(self, image_1, box_1, clearimg_1, image_2, box_2, clearimg_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        new_clearimg = np.array(clearimg_1, np.float32) * 0.5 + np.array(clearimg_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes, new_clearimg


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    clearimg = []
    for i, (img, box, clear) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)
        clearimg.append(clear)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    clearimg = torch.from_numpy(np.array(clearimg)).type(torch.FloatTensor)
    return images, bboxes, clearimg