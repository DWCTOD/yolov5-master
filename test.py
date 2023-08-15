# nms 用于过滤掉一些框
def nms(boxes, scores, iou_threshold):
    # 如滶
    if len(boxes) == 0:
        return []
    # 进行排序
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
