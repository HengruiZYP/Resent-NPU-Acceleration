import cv2
import json
import os
import argparse
import sys

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, parent_path)
# print(sys.path)

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
        "--test_image",
        type=str,
        default="./test_images/ILSVRC2012_val_00000014.jpeg",
        help="Path of test image file.",
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
            - test_image (str): 测试图片路径。
            - with_profile (bool): 是否返回每个阶段的时间消耗。默认为 False。
            - visualize (bool): 是否将结果可视化。默认为 False。
    
    Returns:
        None
    """
    config_file = args.config

    # init PPNCDetector
    assert os.path.exists(config_file), "deploy_config does not exist."
    detector = ResNet(config_file)

    # read the test_image
    test_image = args.test_image
    image = cv2.imread(test_image)

    with_profile = args.with_profile
    if with_profile:
        # return with time consumption for each stage
        results = detector.predict_profile(image)
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
        cv2.imwrite("./vis.jpg", render_img)
        print("visualize result saved as vis.jpg.")


if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    main(args)
