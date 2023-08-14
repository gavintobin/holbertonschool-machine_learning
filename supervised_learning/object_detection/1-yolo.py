#!/usr/bin/env python3
'''task 1'''
from tensorflow import keras as K
import numpy as np


class Yolo():
    '''yolo class'''
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''innit '''
        self.model = K.models.load_model(model_path)
        self.class_names = self.load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def load_classes(self, path):
        '''helper func to load classes'''
        with open(path, 'r') as f:
            classes = f.read().splitlines()
        return classes

    def process_outputs(self, outputs, image_size):
        '''process  image output output'''
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
        # Extract dimensions from output
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Process each anchor box
            for row in range(grid_height):
                for col in range(grid_width):
                    for box in range(anchor_boxes):
                        box_info = output[row, col, box, :]

                        # Extract box coordinates and confidence
                        box_x, box_y, box_w, box_h = box_info[:4]
                        box_confidence = box_info[4]

                        # Calculate box coordinates relative to original image
                        box_x_rel = (col + box_x) / grid_width
                        box_y_rel = (row + box_y) / grid_height
                        box_w_rel = self.anchors[box][0] * np.exp(box_w) / image_size[1]
                        box_h_rel = self.anchors[box][1] * np.exp(box_h) / image_size[0]

                        # Calculate box boundaries
                        x1 = (box_x_rel - box_w_rel / 2) * image_size[1]
                        y1 = (box_y_rel - box_h_rel / 2) * image_size[0]
                        x2 = x1 + box_w_rel * image_size[1]
                        y2 = y1 + box_h_rel * image_size[0]

                        # Append box info to lists
                        boxes.append([x1, y1, x2, y2])
                        box_confidences.append(box_confidence)
                        box_class_probs.append(box_info[5:])

            return boxes, box_confidences, box_class_probs
