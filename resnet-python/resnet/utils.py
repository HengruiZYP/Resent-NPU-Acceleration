"""utils"""
import cv2
import json


with open('./resnet/imagenet_class_index.json','r') as f:
    turn_labels=json.load(f)

def get_label(index):
    return turn_labels[str(index)][1]

def vis(img, results):
    """visualize results"""
    img = img.copy()
    score = results["scores"][0]
    score = round(score, 2)
    label = results["label_ids"][0]
    label = get_label(label)
    text = f"label = {label}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
    return img
