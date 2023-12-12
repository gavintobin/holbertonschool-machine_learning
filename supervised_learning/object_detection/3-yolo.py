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
                        bw = pw * np.exp(tw) / self.model.input.shape[1].value
                        bh = ph * np.exp(th) / self.model.input.shape[2].value

                        x1 = (bx - (bw / 2)).any() * image_width
                        x2 = (bx + (bw / 2)).any() * image_width
                        y2 = (by + (bh / 2)).any() * image_height
                        y1 = (by - (bh / 2)).any() * image_height

                        boxes.append([x1, y1, x2, y2])

        return np.array(boxes), box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        '''filter boxes based on class confidence and threshold'''
        filtered_boxes, box_classes, box_scores = [], [], []

        for box, box_conf, box_class_prob in (zip(
                boxes, box_confidences, box_class_probs)):
            box_scores_per_class = box_conf * box_class_prob
            box_class = np.argmax(box_scores_per_class, axis=-1)
            box_score = np.max(box_scores_per_class, axis=-1)

            mask = box_score >= self.class_t

            filtered_boxes.extend(box[mask])
            box_classes.extend(box_class[mask])
            box_scores.extend(box_score[mask])

        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)

        return filtered_boxes, box_classes, box_scores

    def sigmoid(self, x):
        '''helper func'''
        return 1 / (1 + np.exp(-x))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        '''nonmax suppression'''
        classes = np.unique(box_classes)
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in classes:
            ids = np.where(box_classes == cls)
            keep = filtered_boxes[ids]
            cls_box_scores = box_scores[ids]

            while len(keep) > 0:
                max_score_idx = np.argmax(cls_box_scores)
                box_predictions.append(keep[max_score_idx])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_box_scores[max_score_idx])

                iou_scores = [self.calculate_iou(keep[max_score_idx],
                              box) for box in keep]
                gone = np.where(np.array(iou_scores) > self.nms_t)
                keep = np.delete(keep, gone, axis=0)
                cls_box_scores = np.delete(cls_box_scores, gone, axis=0)

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    def calculate_iou(self, box1, boxes2):
        '''intersect over union'''
        x1 = np.maximum(box1[0], boxes2[0])
        y1 = np.maximum(box1[1], boxes2[1])
        x2 = np.minimum(box1[2], boxes2[2])
        y2 = np.minimum(box1[3], boxes2[3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes2_area = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        union = box1_area + boxes2_area - intersection

        iou = intersection / np.maximum(union, np.finfo(float).eps)

        return iou
