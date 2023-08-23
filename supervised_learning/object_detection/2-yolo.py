
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
                mask = class_scores >= self.class_threshold
                filtered_boxes.extend(box[mask])
                box_classes.extend(class_indices[mask])
                box_scores.extend(class_scores[mask])

            filtered_boxes = np.array(filtered_boxes)
            box_classes = np.array(box_classes)

            box_scores = np.array(box_scores)

            return filtered_boxes, box_classes, box_scores



    def sigmoid(self, x):
        """
        Sigmoid Function
        """
        return (1 / (1 + np.exp(-x)))

