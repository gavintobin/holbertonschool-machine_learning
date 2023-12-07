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
        for output_boxes, box_confidence, box_class_prob in zip(boxes, box_confidences, box_class_probs):
            grid_height, grid_width, anchors = output_boxes.shape[:3]

            for cy in range(grid_height):
                for cx in range(grid_width):
                    for b in range(anchors):
                        confidence = float(box_confidence[cy, cx, b])
                        class_probs = box_class_prob[cy, cx, b, :]

                        if confidence > self.class_t:
                            pw, ph = self.anchors[b][:2]
                            tx, ty, tw, th = output_boxes[cy, cx, b]
                            bx = (self.sigmoid(tx) + cx) / grid_width
                            by = (self.sigmoid(ty) + cy) / grid_height
                            bw = pw * np.exp(tw) / self.model.input.shape[1]
                            bh = ph * np.exp(th) / self.model.input.shape[2]

                            x1 = max(int((bx - (bw / 2)) * image_width), 0)
                            y1 = max(int((by - (bh / 2)) * image_height), 0)
                            x2 = min(int((bx + (bw / 2)) * image_width), image_width)
                            y2 = min(int((by + (bh / 2)) * image_height), image_height)

                            all_boxes.append([x1, y1, x2, y2])

        return np.array(all_boxes)

    def sigmoid(self, x):
        '''helper func'''
        return 1 / (1 + np.exp(-x))
