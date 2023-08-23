#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
import tensorflow.keras as K
import cv2
import os

class Yolo:
    """
    Yolo Class
    """
    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """
        Class Constructor
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path) as file:
            class_names = file.read()

        self.class_names = class_names.replace("\n", "|").split("|")[:-1]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def preprocess_images(self, images):
        """
        Preprocess Images
        """

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])

            resized_image = cv2.resize(image,
                                       (self.model.input.shape[1],
                                        self.model.input.shape[2]),
                                       interpolation=cv2.INTER_CUBIC)
            rescaled_image = resized_image / 255

            pimages.append(rescaled_image)

        pimages = np.asarray(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    @staticmethod
    def load_images(folder_path):
        """
        Loading Images
        """
        img_pth = []
        img = []
        for image in os.listdir(folder_path):
            img.append(cv2.imread(folder_path + '/' + image))
            img_pth.append(folder_path + '/' + image)

        return img, img_pth

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Non-Max Suppression
        """
        selected_boxes = []
        selected_classes = []
        selected_scores = []

        for c in set(box_classes):
            i = np.where(box_classes == c)
            cls_boxes = filtered_boxes[i]
            cls_box_scores = box_scores[i]

            while len(cls_boxes) > 0:
                max_idx = np.argmax(cls_box_scores)
                selected_boxes.append(cls_boxes[max_idx])
                selected_classes.append(c)
                selected_scores.append(cls_box_scores[max_idx])

                cls_boxes = np.delete(cls_boxes, max_idx, axis=0)
                cls_box_scores = np.delete(cls_box_scores, max_idx, axis=0)

                if len(cls_boxes) == 0:
                    break

                iou = self.calculate_iou(selected_boxes[-1], cls_boxes)
                mask = iou < self.nms_t

                cls_boxes = cls_boxes[mask]
                cls_box_scores = cls_box_scores[mask]

        selected_boxes = np.array(selected_boxes)
        selected_classes = np.array(selected_classes)
        selected_scores = np.array(selected_scores)

        return selected_boxes, selected_classes, selected_scores

    def calculate_iou(self, box, boxes):
        """
        Intersection over union
        """
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersect = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return intersect / (area + boxes_area - intersect)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter Boxes
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box in range(len(boxes)):
            box_classes.append(np.argmax(box_class_probs[box]
                                         * box_confidences[box],
                                         axis=-1).reshape(-1))
            box_scores.append(np.max(box_class_probs[box]
                                     * box_confidences[box],
                                     axis=-1).reshape(-1))

        box_classes_con = np.concatenate(box_classes)
        box_scores_con = np.concatenate(box_scores)
        mask = box_scores_con >= self.class_t

        filtered_boxes = np.concatenate(
            [box.reshape(-1, 4) for box in boxes], axis=0)
        filtered_boxes = filtered_boxes[mask]

        box_classes = box_classes_con[mask]
        box_scores = box_scores_con[mask]

        return filtered_boxes, box_classes, box_scores

    def process_outputs(self, outputs, image_size):
        """
        Process Outputs
        """
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

    def sigmoid(self, x):
        """
        Sigmoid Function
        """
        return (1 / (1 + np.exp(-x)))
