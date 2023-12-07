#!/usr/bin/env python3
'''task 1'''
import numpy as np
from tensorflow.keras.models import load_model


class Yolo:
    '''yolo class'''
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''init yolo class'''
        self.model = load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        '''process outputs'''
        boxes, box_confidences, box_class_probs = [], [], []
        image_height, image_width = image_size

        for output in outputs:
            boxes.append(output[..., :4])
            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        all_boxes = []
        for output in range(len(boxes)):
            grid_height = outputs[output].shape[0]
            grid_width = outputs[output].shape[1]
            anchors = outputs[output].shape[2]

            for cy in range(grid_height):
                for cx in range(grid_width):
                    for b in range(anchors):
                        pw, ph = self.anchors[b][:2]
                        tx, ty, tw, th = boxes[output][cy, cx, b]
                        bx = (self.sigmoid(tx) + cx) / grid_width
                        by = (self.sigmoid(ty) + cy) / grid_height
                        bw = pw * np.exp(tw) / self.model.input.shape[1].value
                        bh = ph * np.exp(th) / self.model.input.shape[2].value

                        x1 = max(int((bx - (bw / 2)).any() * image_width), 0)
                        x2 = min(int((bx + (bw / 2)).any() * image_width),
                                 image_width)
                        y2 = min(int((by + (bh / 2)).any() * image_height),
                                 image_height)
                        y1 = max(int((by - (bh / 2)).any() * image_height), 0)

                        all_boxes.append([x1, y1, x2, y2])

        return np.array(all_boxes), box_confidences, box_class_probs

    def sigmoid(self, x):
        '''helper func'''
        return 1 / (1 + np.exp(-x))
