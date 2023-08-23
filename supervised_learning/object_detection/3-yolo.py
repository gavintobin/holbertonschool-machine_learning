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
        '''process output'''
        boxes, box_confidences, box_class_probs = [], [], []
        image_height, image_width = image_size

        for output in range(len(outputs)):
            boxes.append(outputs[output][..., :4])
            box_confidences.append(self.sigmoid(outputs[output][..., 4:5]))
            box_class_probs.append(self.sigmoid(outputs[output][..., 5:]))

        for output in range(len(boxes)):
            grid_height = outputs[output].shape[0]
            grid_width = outputs[output].shape[1]
            anchors = outputs[output].shape[2]

            for cy in range(grid_height):
                for cx in range(grid_width):
                    for b in range(anchors):
                        tx, ty, tw, th = boxes[output][cy, cx, b]
                        pw, ph = self.anchors[output][b]
                        bx = (self.sigmoid(tx)) + cx
                        by = (self.sigmoid(ty)) + cy
                        bw = pw * np.exp(tw)
                        bh = ph * np.exp(th)
                        bx /= grid_width
                        by /= grid_height
                        bw /= self.model.input.shape[1].value
                        bh /= self.model.input.shape[2].value
                        x1 = (bx - (bw / 2)) * image_width
                        y1 = (by - (bh / 2)) * image_height
                        x2 = (bx + (bw / 2)) * image_width
                        y2 = (by + (bh / 2)) * image_height
                        boxes[output][cy, cx, b] = [x1, y1, x2, y2]

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        '''filter boxes'''
        filtered_boxes, box_classes, box_scores = [], [], []

        for box, confidence, class_probs in zip(boxes, box_confidences, box_class_probs):
            # Calculate box scores by multiplying box_confidence and class probabilities
            scores = confidence * class_probs

            # Find indices of class predictions that exceed class threshold
            class_indices = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            # Filter out boxes with scores below box threshold
            mask = class_scores >= self.class_t
            filtered_boxes.extend(box[mask])
            box_classes.extend(class_indices[mask])
            box_scores.extend(class_scores[mask])

        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)

        box_scores = np.array(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        sorted_indices = np.argsort(box_scores)[::-1]
        selected_indices = []
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        while sorted_indices.size > 0:
            highest_score_idx = sorted_indices[0]
            selected_indices.append(highest_score_idx)

            ious = self.calculate_iou(filtered_boxes[highest_score_idx], filtered_boxes[sorted_indices[1:]])
            filtered_indices = np.where(ious <= self.nms_t)[0]
            sorted_indices = sorted_indices[filtered_indices + 1]

        box_predictions = filtered_boxes[selected_indices]
        predicted_box_classes = box_classes[selected_indices]
        predicted_box_scores = box_scores[selected_indices]

        return box_predictions, predicted_box_classes, predicted_box_scores

    def calculate_iou(self, box1, boxes):
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        iou = intersection_area / (box1_area + boxes_area - intersection_area)
        return iou

    def sigmoid(self, x):
        """
        Sigmoid Function
        """
        return (1 / (1 + np.exp(-x)))

