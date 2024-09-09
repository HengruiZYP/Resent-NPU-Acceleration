import argparse
import os
import sys
import cv2
import json
import logging
import numpy as np
import paddle
from paddle import fluid
import warnings
import pickle
import time
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, parent_path)

from resnet import ResNet, vis

warnings.filterwarnings("ignore")

log_formatter = "%(levelname)s %(asctime)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_formatter)


class PaddleInfer(ResNet):
    """PaddleInfer"""
    def __init__(self, model_dir, config):
        """
        Args:
            model_dir (str): model path
            config (dict): deploy_config
        """
        super().__init__(config)
        self.model_dir = model_dir
        self.model_file = None
        self.params_file = None
        for file in os.listdir(model_dir):
            if file.endswith(".pdmodel"):
                self.model_file = file
            elif file.endswith(".pdiparams"):
                self.params_file = file
        assert self.model_file is not None, "pdmodel file does not exsit."
        assert self.params_file is not None, "pdiparams file does not exist."

    def load(self):
        """load model"""
        self.exe = fluid.Executor(fluid.CPUPlace())
        [self.paddle_prog, feed, self.fetch] = fluid.io.load_inference_model(
            self.model_dir,
            self.exe,
            model_filename=self.model_file,
            params_filename=self.params_file,
        )

    def predict(self, input_dict):
        """infer"""
        res = self.exe.run(self.paddle_prog, feed=input_dict, fetch_list=self.fetch)
        return res[0]

    def predict_image(self, img):
        """predict a image with preprocess and postprocess"""
        """ " predict image"""
        inputs = self.preprocessor(img)
        inputs = {"x": inputs}
        res = self.predict(inputs)
        res = self.postprocessor(res)
        return res

def argsparser():
    """
    parse command arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="./model/config.json",
        help=("path of deploy config.json"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model",
        help=("path of pdmodel and pdiparams"),
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./test_images",
        help="Dir of test image file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output_dir", help="output dir."
    )
    return parser


def main(args):
    """main"""
    
    # get test_images
    test_dir = args.test_dir
    assert test_dir is not None, "test_dir must be provided."
    assert os.path.exists(test_dir), "test_dir does not exist."

    # default config
    
    config_file = args.config
    assert os.path.exists(config_file), "config_file does not exist."

    # check output_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ppnc_output_image_dir = os.path.join(output_dir, "ppnc_result_images")
    if not os.path.exists(ppnc_output_image_dir):
        os.makedirs(ppnc_output_image_dir)
    ppnc_output_pickle_dir = os.path.join(output_dir, "ppnc_result_pickle")
    if not os.path.exists(ppnc_output_pickle_dir):
        os.makedirs(ppnc_output_pickle_dir)

    paddle_output_image_dir = os.path.join(output_dir, "paddle_result_images")
    if not os.path.exists(paddle_output_image_dir):
        os.makedirs(paddle_output_image_dir)
    paddle_output_pickle_dir = os.path.join(output_dir, "paddle_result_pickle")
    if not os.path.exists(paddle_output_pickle_dir):
        os.makedirs(paddle_output_pickle_dir)

    # initial PPNC Detector
    assert os.path.exists(config_file), "config does not exist."

    # initial paddle Detector
    model_dir = args.model_dir
    assert os.path.exists(model_dir), "model does not exist."

    # create detector
    ppnc_infer = ResNet(config_file)
    paddle_infer = PaddleInfer(model_dir, config_file)
    paddle_infer.load()

    # infer
    for image_file in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_file)
        print(image_file)
        img = cv2.imread(image_path)




        start_time_ppnc = time.time()
        ppnc_res = ppnc_infer.predict_image(img)
        end_time_ppnc = time.time()
        ppnc_time = end_time_ppnc - start_time_ppnc

        start_time_paddle = time.time()
        paddle_res = paddle_infer.predict_image(img)
        end_time_paddle = time.time()
        paddle_time = end_time_paddle - start_time_paddle

        print(f'ppnc   time: {ppnc_time}')
        print(f'Paddle time: {paddle_time}\n')

        ppnc_original = ppnc_res["original"]
        paddle_original = paddle_res["original"]

        # save visualize result
        ppnc_vis = vis(img, ppnc_res)
        paddle_vis = vis(img, paddle_res)
        cv2.imwrite(os.path.join(paddle_output_image_dir, image_file), paddle_vis)
        cv2.imwrite(os.path.join(ppnc_output_image_dir, image_file), ppnc_vis)

        diff = np.linalg.norm(np.array(ppnc_original) - np.array(paddle_original))

        # save pickle result
        with open(
            os.path.join(ppnc_output_pickle_dir, image_file.split(".")[0] + ".pkl"),
            "wb",
        ) as ppnc_pickle_file:
            pickle.dump(ppnc_original, ppnc_pickle_file)

        with open(
            os.path.join(paddle_output_pickle_dir, image_file.split(".")[0] + ".pkl"),
            "wb",
        ) as paddle_pickle_file:
            pickle.dump(paddle_original, paddle_pickle_file)



if __name__ == "__main__":
    paddle.enable_static()
    parser = argsparser()
    args = parser.parse_args()
    main(args)
