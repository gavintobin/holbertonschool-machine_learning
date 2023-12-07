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
                        pw, ph = self.anchors[output][b]
                        tx, ty, tw, th = boxes[output][cy, cx, b]
                        bx = (self.sigmoid(tx)) + cx
                        by = (self.sigmoid(ty)) + cy
                        bw = pw * np.exp(tw) / self.model.input.shape[1]
                        bh = ph * np.exp(th) / self.model.input.shape[2]

                        x1 = (bx - (bw / 2)).any() * image_width
                        x2 = (bx + (bw / 2)).any() * image_width
                        y2 = (by + (bh / 2)).any() * image_height
                        y1 = (by - (bh / 2)).any() * image_height

                        boxes.append([x1, y1, x2, y2])

        return np.array(boxes), box_confidences, box_class_probs

        def filter_boxes(self, boxes, box_confidences, box_class_probs):
            '''filter boxes'''
            filtered_boxes, box_classes, box_scores = [], [], []

            for i in range(len(boxes)):
                box = boxes[i]
                box_confidence = box_confidences[i]
                box_class_prob = box_class_probs[i]

                box_scores.extend(np.max(box_class_prob, axis=-1)
                                  * box_confidence)
                box_classes.extend(np.argmax(box_class_prob, axis=-1))
                filtered_boxes.extend(box)

            filtered_boxes = np.array(filtered_boxes)
            box_classes = np.array(box_classes)
            box_scores = np.array(box_scores)

            # Filter boxes based on box scores
            mask = box_scores >= self.class_t
            filtered_boxes = filtered_boxes[mask]
            box_classes = box_classes[mask]
            box_scores = box_scores[mask]

            return filtered_boxes, box_classes, box_scores

    def sigmoid(self, x):
        '''helper func'''
        return 1 / (1 + np.exp(-x))
