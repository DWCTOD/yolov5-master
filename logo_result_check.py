import os
import cv2
import shutil
from glob import glob

def iou(rect1, rect2):
    xmin1, ymin1, xmax1, ymax1 = rect1
    xmin2, ymin2, xmax2, ymax2 = rect2
    w1 = xmax1 - xmin1
    h1 = ymax1 - ymin1
    w2 = xmax2 - xmin2
    h2 = ymax2 - ymin2
    xmin_inter = max(xmin1, xmin2)
    ymin_inter = max(ymin1, ymin2)
    xmax_inter = min(xmax1, xmax2)
    ymax_inter = min(ymax1, ymax2)
    inter = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)
    union = w1 * h1 + w2 * h2 - inter
    if inter <= 0:
        return 0
    else:
        return inter / union

def get_txt_info(txt_path):
    label_info = []
    with open(txt_path, "r") as inf:
        label_content = inf.readlines()
    for i in label_content:
        tmp_info = i.strip().split(" ")
        label_info.append(tmp_info)
    print(txt_path)
    print(label_info)
    return label_info

def yolo2labelme(bbox,img_height, img_width):
    x1,y1,w,h = bbox
    x_min = int(x1 * img_width - (w * img_width) / 2)
    y_min = int(y1 * img_height - (h * img_height) / 2)
    x_max = int(x1 * img_width + (w * img_width) / 2)
    y_max = int(y1 * img_height + (h * img_height) / 2)

    return x_min, y_min, x_max, y_max

def check_one(test_label, predict_label,img_path):
    pred_label_info = get_txt_info(predict_label)
    try:
        ori_label_info = get_txt_info(test_label)
    except:
        print("该标签不在测试数据集中")
        return True
    total_label_num = len(ori_label_info)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    count = 0
    for tmp_info in pred_label_info:
        l_index = tmp_info[0]
        l_bbox = [float(i) for i in tmp_info[1:]]
        l_bbox = yolo2labelme(l_bbox, h,w)

        for ori_info in ori_label_info:
            ori_l_index = ori_info[0]
            ori_l_bbox = [float(i) for i in ori_info[1:]]
            ori_l_bbox = yolo2labelme(ori_l_bbox, h, w)
            iou_score = iou(l_bbox, ori_l_bbox)
            if iou_score < 0.85:
                continue
            if l_index == ori_l_index:
                print(tmp_info)
                count += 1
    if count == total_label_num:
        return False # 预测和标注结果一致,不移动
    else:
        return True


def check_copy(test_path, predict_path, out_path):
    # test_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('txt')]
    predict_files = [os.path.join(predict_path, f) for f in os.listdir(predict_path) if f.endswith('txt')]
    for predict_file in predict_files:
        txt_file_name = os.path.basename(predict_file)
        # if txt_file_name != "63d11d70000000001c0366c5_312.txt":
        #     continue
        test_file = os.path.join(test_path,txt_file_name)
        # prefix = ".".join(test_file.split(".")[:-1])
        # img_path = [i for i in glob(f"{prefix}*") if (not i.endswith('txt') and not i.endswith('json'))][0]
        prefix = "/".join(predict_file.split("/")[:-2])
        tmp_name = txt_file_name.split(".")[0]
        img_path = [i for i in glob(f"{prefix}/{tmp_name}*") if (not i.endswith('txt') and not i.endswith('json'))][0]

        Flag = check_one(test_file, predict_file, img_path) # True 时 表示不一致
        print(Flag)
        print(img_path)
        if Flag:
            try:
                shutil.copy(test_file, out_path)
                shutil.copy(img_path, out_path)
            except:
                continue




if __name__ =="__main__":
    # TODO 修改输入的参数
    dataset_path = "第四轮"
    detect_path = "exp12"
    root_path = "/data/panzhiwei/NIKE/NIKE_dataset/TODO_val_dataset/return_dataset"
    test_path = f"/data/panzhiwei/NIKE/NIKE_dataset/ifashion/val_dataset/{dataset_path}/labels/val2017"

    predict_path = f"/data/panzhiwei/NIKE_deployment/Cloth-Yolov5/yolov5-master/runs/detect/{detect_path}/labels/"
    out_path = f"{root_path}/checked_dataset/{dataset_path}/out" # 将不一致的结果存储到这里
    os.makedirs(out_path, exist_ok=True)
    check_copy(test_path, predict_path, out_path)

    # label = "/data/panzhiwei/NIKE_deployment/Cloth-Yolov5/yolov5-master/runs/val/exp53/labels/63d7b3d4000000001a02502b_179.txt"
    # ori_label = "/data/panzhiwei/NIKE/NIKE_dataset/TODO_val_dataset/return_dataset/第一轮/train/63d7b3d4000000001a02502b_179.txt"
    #


    # label_info = get_txt_info(label)
    # print(label_info)
    # ori_label_info = get_txt_info(ori_label)
    # print(ori_label_info)
    # # img_path = "/data/panzhiwei/NIKE/NIKE_dataset/TODO_val_dataset/return_dataset/第一轮/train/63d7b3d4000000001a02502b_179.jpeg"
    # prefix = ".".join(ori_label.split(".")[:-1])
    # img_path = [i for i in glob(f"{prefix}*") if (not i.endswith('txt') and not i.endswith('json'))][0]
    # img = cv2.imread(img_path)
    # h,w = img.shape[:2]
    # for tmp_info in label_info:
    #     l_index = tmp_info[0]
    #     l_bbox = [float(i) for i in tmp_info[1:]]
    #     l_bbox = yolo2labelme(l_bbox, h,w)
    #
    #     for ori_info in ori_label_info:
    #         ori_l_index = ori_info[0]
    #         ori_l_bbox = [float(i) for i in ori_info[1:]]
    #         ori_l_bbox = yolo2labelme(ori_l_bbox, h, w)
    #         iou_score = iou(l_bbox, ori_l_bbox)
    #         if iou_score < 0.9:
    #             continue
    #         if l_index == ori_l_index:
    #             print(tmp_info)


