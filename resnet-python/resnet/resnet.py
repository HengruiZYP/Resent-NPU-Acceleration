"""detector definition"""
import time
import cv2
from .preprocess import Preprocess
from .postprocess import Postprocess
import numpy as np
import os, sys
upper_level_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lib"))
sys.path.append(upper_level_path)
from libresnet_bind import ResnetPredictor


class ResNet(object):
    """ResNet class."""

    def __init__(self, config, target_size=[224, 224]):
        """init"""
        self.model = ResnetPredictor(config)
        self.preprocessor = Preprocess(target_size)
        self.postprocessor = Postprocess()
        self.preprocess_time = 0
        self.predict_time = 0
        self.postprocess_time = 0


    def predict_image(self, img):
        """ " predict image"""
        inputs = self.preprocessor(img)
        res = self.model.predict(inputs)
        res = self.postprocessor(res)
        return res

    def predict_profile(self, img):
        """predict image with profile"""
        time0 = time.time()
        inputs = self.preprocessor(img)
        time1 = time.time()
        self.preprocess_time = time1 - time0

        res = self.model.predict(inputs)
        time2 = time.time()
        self.predict_time = time2 - time1

        results = self.postprocessor(res)
        time3 = time.time()
        self.postprocess_time = time3 - time2
        return results