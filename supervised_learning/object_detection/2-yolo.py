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

        for output, output_boxes in enumerate(boxes):
            grid_height, grid_width, anchors = output_boxes.shape[:3]

        for cy in range(grid_height):
            for cx in range(grid_width):
                for b in range(anchors):
                    tx, ty, tw, th = output_boxes[cy, cx, b]
                    pw, ph = self.anchors[output][b]
                    bx = (self.sigmoid(tx) + cx) / grid_width
                    by = (self.sigmoid(ty) + cy) / grid_height
                    bw = pw * np.exp(tw) / self.model.input.shape[1].value
                    bh = ph * np.exp(th) / self.model.input.shape[2].value
                    x1 = (bx - (bw / 2)) * image_width
                    y1 = (by - (bh / 2)) * image_height
                    x2 = (bx + (bw / 2)) * image_width
                    y2 = (by + (bh / 2)) * image_height
                    output_boxes[cy, cx, b] = [x1, y1, x2, y2]

        return boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        '''helper func'''
        return 1 / (1 + np.exp(-x))

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        '''filter boxes'''
        all_boxes = []
        all_classes = []
        all_scores = []

        # Iterate through each output scale
        for i in range(len(boxes)):
            grid_height, grid_width, anchor_boxes, _ = boxes[i].shape

            # Reshape boxes, box_confidences, and box_class_probs
            boxes_reshaped = boxes[i].reshape((grid_height * grid_width * anchor_boxes, 4))
            confidences_reshaped = box_confidences[i].reshape((grid_height * grid_width * anchor_boxes,))
            class_probs_reshaped = box_class_probs[i].reshape((grid_height * grid_width * anchor_boxes, len(self.class_names)))

            # Filter out boxes with low object confidence
            mask_conf = confidences_reshaped >= self.class_threshold
            boxes_filtered = boxes_reshaped[mask_conf]
            confidences_filtered = confidences_reshaped[mask_conf]
            class_probs_filtered = class_probs_reshaped[mask_conf]

            # Calculate box scores
            box_scores = confidences_filtered * class_probs_filtered.max(axis=1)

            mask_scores = box_scores > 0
            boxes_filtered = boxes_filtered[mask_scores]
            box_scores = box_scores[mask_scores]
            classes_filtered = class_probs_filtered.argmax(axis=1)[mask_scores]

            all_boxes.append(boxes_filtered)
            all_classes.append(classes_filtered)
            all_scores.append(box_scores)

        # Concatenate results from all scales
        if all_boxes:
            filtered_boxes = np.concatenate(all_boxes, axis=0)
            box_classes = np.concatenate(all_classes, axis=0)
            box_scores = np.concatenate(all_scores, axis=0)
        else:
            filtered_boxes = np.array([])
            box_classes = np.array([])
            box_scores = np.array([])

        return filtered_boxes, box_classes, box_scores
