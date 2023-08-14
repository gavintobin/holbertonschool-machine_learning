#!/usr/bin/env python3
'''task 1'''
from tensorflow import keras as K


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
