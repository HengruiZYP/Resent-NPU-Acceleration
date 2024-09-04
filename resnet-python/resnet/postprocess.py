"""postprocess of picodet"""
import numpy as np
import cv2


class Postprocess(object):
    """postprocess of resnet"""

    def __init__(self, topk=5):
        """init"""
        self.topk = topk

    def __call__(self, result):
        """返回指定topk的标签id和对应的得分，并附带原始结果。
        
        Args：
            result (numpy.ndarray): 一个形状为（batch size, num class）的二维数组，
                表示模型在每个样本上的预测分类分数。
        
        Returns：
            dict: 包含以下键值对的字典：
        
                - "label_ids": list[int]: 指定topk的标签id列表。
                - "scores": list[float]: 与上述标签id对应的预测分数列表。
                - "original": numpy.ndarray: 输入的原始预测结果。
        
        """
        original = result
        result = np.squeeze(result)
        index = result.argsort(axis=0)[-self.topk: ][::-1]
        label_ids = index.tolist()
        scores = []
        for i in label_ids:
            scores.append(result[i].item())

        res = {}
        res["label_ids"] = label_ids
        res["scores"] = scores
        res["original"] = original
        return res
