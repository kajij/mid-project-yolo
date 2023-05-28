import cv2
import numpy as np
import sys
import os
sys.path.append("./darknet_build/Release")
import ctypes
darknet = ctypes.cdll.LoadLibrary("./darknet_build/Release/darknet.dll")
import darknet
def load_yolo_model(config_path, weights_path):
    """
    加载YOLO模型
    :param config_path: 模型配置文件路径
    :param weights_path: 模型权重文件路径
    :return: 模型
    """
    net = darknet.load_net_custom(config_path.encode("ascii"), weights_path.encode("ascii"), 0)
    meta = darknet.load_meta("C:/ml/transportation.data".encode("ascii"))
    return net, meta

def preprocess_image(image):
    """
    将图像转换为Darknet格式
    :param image: 原始图像
    :return: Darknet格式的图像
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    darknet_image = darknet.make_image(image.shape[1], image.shape[0], 3)
    darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())
    return darknet_image

def postprocess_detections(detections):
    """
    后处理预测结果，提取边界框坐标、类别和置信度
    :param detections: 预测结果
    :return: 边界框列表 [x1, y1, x2, y2, class_name, confidence]
    """
    predicted_boxes = []
    for detection in detections:
        class_name = detection[0].decode()
        confidence = detection[1]
        x, y, w, h = detection[2]
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        predicted_boxes.append([x1, y1, x2, y2, class_name, confidence])
    return predicted_boxes

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU（重叠联合）
    :param box1: 边界框1的坐标 [x1, y1, x2, y2]
    :param box2: 边界框2的坐标 [x1, y1, x2, y2]
    :return: IoU值
    """
    # 计算边界框的坐标
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # 计算交集的坐标
    intersect_x1 = max(x1, x3)
    intersect_y1 = max(y1, y3)
    intersect_x2 = min(x2, x4)
    intersect_y2 = min(y2, y4)

    # 计算交集的面积
    intersect_area = max(0, intersect_x2 - intersect_x1 + 1) * max(0, intersect_y2 - intersect_y1 + 1)

    # 计算并集的面积
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = box1_area + box2_area - intersect_area

    # 计算IoU
    iou = intersect_area / union_area

    return iou

def calculate_metrics(predicted_boxes, true_boxes, iou_threshold):
    """
    计算模型的准确率、召回率和平均精度（mAP）
    :param predicted_boxes: 预测的边界框列表，每个边界框的坐标 [x1, y1, x2, y2]
    :param true_boxes: 真实的边界框列表，每个边界框的坐标 [x1, y1, x2, y2]
    :param iou_threshold: 用于判断是否正确检测到目标的IoU阈值
    :return: 准确率、召回率和mAP
    """
    num_predicted = len(predicted_boxes)
    num_true = len(true_boxes)
    num_true_positive = 0
    average_precision = 0.0

    # 记录每个预测边界框的匹配情况
    true_positive = np.zeros(num_predicted)

    for i in range(num_true):
        best_iou = 0.0
        best_idx = -1

        for j in range(num_predicted):
            iou = calculate_iou(predicted_boxes[j], true_boxes[i])
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        # 判断是否正确检测到目标
        if best_iou >= iou_threshold:
            if true_positive[best_idx] == 0:
                # 正确检测到目标
                true_positive[best_idx] = 1
                num_true_positive += 1
        else:
            # 未能检测到目标
            pass

    # 计算准确率、召回率和mAP
    precision = num_true_positive / num_predicted
    recall = num_true_positive / num_true

    # 计算平均精度（mAP）
    sorted_indices = np.argsort(-np.array(predicted_scores))  # 根据预测分数进行排序，假设预测分数存储在predicted_scores列表中
    true_positive_cumsum = np.cumsum(true_positive[sorted_indices])
    precision_cumsum = true_positive_cumsum / np.arange(1, num_predicted + 1)
    recall_cumsum = true_positive_cumsum / num_true
    interpolated_precision = np.maximum.accumulate(precision_cumsum[::-1])[::-1]
    average_precision = np.mean(interpolated_precision)

    return precision, recall, average_precision


# 加载YOLO模型
net, meta = load_yolo_model("C:/ml/yolov4-tiny-custom.cfg", "C:/ml/backup/yolov4-tiny-custom_last.weights")

# 读取图像
image = cv2.imread("C:/ml/test1.jpg")

# 将图像转换为Darknet格式
darknet_image = preprocess_image(image)

# 进行预测
detections = darknet.detect_image(net, meta, darknet_image)

# 提取预测边界框的坐标和置信度
predicted_boxes = postprocess_detections(detections)

# 打印预测边界框
for box in predicted_boxes:
    x1, y1, x2, y2, class_name, confidence = box
    print("类别：", class_name)
    print("置信度：", confidence)
    print("边界框坐标：", x1, y1, x2, y2)

# 释放内存
darknet.free_image(darknet_image)

# 真实边界框
true_boxes = [[x1, y1, x2, y2], ...]  # 根据实际情况填写真实边界框的坐标

# 计算指标
iou_threshold = 0.5  # IoU阈值
precision, recall, average_precision = calculate_metrics(predicted_boxes, true_boxes, iou_threshold)

# 打印指标
print("准确率：", precision)
print("召回率：", recall)
print("mAP：", average_precision)
