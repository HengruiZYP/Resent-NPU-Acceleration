import cv2
import json
import os
import argparse
import sys
import time
from pathlib import Path

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, parent_path)

from resnet import ResNet, vis


def argsparser():
    """
    解析命令行参数，配置模型参数和优化器等参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="./model/config.json",
        help=("path of deploy config.json"),
    )
    parser.add_argument(
        "--test_folder",
        type=str,
        default="./test_images",
        help="Path of test folder file.",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="whether to visualize."
    )
    parser.add_argument(
        "--with_profile", action="store_true", help="whether to predict with profile."
    )
    return parser


def main(args):
    """
    根据输入的配置参数和测试图片，使用预训练模型进行检测。并可对结果进行可视化。
    
    Args:
        args ([type]): 包含配置信息的参数类对象。包含以下字段：
            - config (str): 配置文件路径。
            - test_folder (str): 测试图片folder路径。
            - with_profile (bool): 是否返回每个阶段的时间消耗。默认为 False。
            - visualize (bool): 是否将结果可视化。默认为 False。
    
    Returns:
        None
    """
    config_file = args.config

    # init PPNCDetector
    assert os.path.exists(config_file), "deploy_config does not exist."
    detector = ResNet(config_file)
    
    # read the test_folder
    test_folder = args.test_folder
    assert os.path.exists(test_folder), "test_folder does not exist."
    
    # init the res_folder
    res_folder = "./res_folder"    
    os.makedirs(res_folder, exist_ok=True)
    
    for image_file in Path(test_folder).glob("*.jpeg"):        
        image_path = str(image_file)       
        image = cv2.imread(image_path)        
        with_profile = args.with_profile
        if with_profile:
            # return with time consumption for each stage
            results = detector.predict_profile(image)
            print("image: ",image_path)
            print("preprocess time: ", detector.preprocess_time)
            print("predict time: ", detector.predict_time)
            print("postprocess time:", detector.postprocess_time)

            total_time = (
                detector.preprocess_time + detector.predict_time + detector.postprocess_time
            )
            print("total time: ", total_time)
        else:
            results = detector.predict_image(image)
        
        visualize = args.visualize
        if visualize:
            render_img = vis(image, results)
            output_path = os.path.join(res_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, render_img)
            print(f"Visualize result saved as {output_path}.")


if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    main(args)
