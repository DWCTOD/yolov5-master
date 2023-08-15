import os
from utils.plots import Annotator, colors, save_one_box
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import shutil

def yolo2labelme(bbox,img_height, img_width):
    x1,y1,w,h = bbox
    x_min = int(x1 * img_width - (w * img_width) / 2)
    y_min = int(y1 * img_height - (h * img_height) / 2)
    x_max = int(x1 * img_width + (w * img_width) / 2)
    y_max = int(y1 * img_height + (h * img_height) / 2)

    return x_min, y_min, x_max, y_max


def get_txt_info(txt_path):
    label_info = []
    with open(txt_path, "r") as inf:
        label_content = inf.readlines()
    for i in label_content:
        tmp_info = i.strip().split(" ")
        label_info.append(tmp_info)
    return label_info

def draw_box_label(ori_im, im, box, label,color=(255,0,0), txt_color=(255,0,0), save_path=None):
    # im = Image.open(img_path)
    # w = im.width  # 图片的宽
    # h = im.height
    # box = yolo2labelme(box,h, w)
    size = max(round(sum(im.size) / 2 * 0.035), 12)
    print(box)
    font = ImageFont.truetype('SimHei.ttf', size)
    # lw = max(round(sum(im.shape) / 2 * 0.003), 2)
    draw = ImageDraw.Draw(im)
    draw.text((box[0], box[1]), label,txt_color, font=font)
    # draw.text((box[0], box[1]), label, txt_color)
    # draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=font)
    draw.rectangle(box, outline=color)
    size1, size2 =ori_im.size, im.size
    combine_result = Image.new("RGB", (size1[0]+size2[0], size1[1]))
    loc1, loc2 = (0, 0), (size1[0], 0)
    combine_result.paste(ori_im, loc1)
    combine_result.paste(im, loc2)
    # im.save('new_test_2.jpg')
    combine_result.save(save_path)
if __name__ == "__main__":
    line_thickness = 3
    names = "test"
    label_list = ["卫衣/套头衫", "短袖T恤", "POLO衫", "运动内衣/背心", "冲锋衣", "棉服/夹克", "皮肤衣/防晒衣",
           "其他上装", "leggings/紧身裤", "运动短裤", "运动裙", "其他下装", "篮球鞋", "跑步鞋", "板鞋",
           "休闲鞋", "帆布鞋","拖鞋/凉鞋/沙滩鞋", "其他鞋子"]

    # TODO 修改输入的参数
    dataset_path = "第四轮"
    detect_path = "exp11"
    root_path = "/data/panzhiwei/NIKE/NIKE_dataset/TODO_val_dataset/return_dataset"
    txt_dir = f"{root_path}/checked_dataset/{dataset_path}/out/"
    imgs_dir = f"{root_path}/checked_dataset/{dataset_path}/imgs"
    predict_path = f"/data/panzhiwei/NIKE_deployment/Cloth-Yolov5/yolov5-master/runs/detect/{detect_path}"
    os.makedirs(imgs_dir, exist_ok=True)
    img_files = [os.path.join(predict_path, img) for img in os.listdir(predict_path) if img.endswith("jpeg")]
    for img_file in img_files:
        shutil.copy(img_file, imgs_dir)
    visualize_path = f"{root_path}/checked_dataset/{dataset_path}/visualize"
    os.makedirs(visualize_path, exist_ok=True)
    txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('txt')]
    for txt_file in tqdm(txt_files):
        txt_name = os.path.basename(txt_file)
        # if txt_name != "63d11d70000000001c0366c5_312.txt":
        #     continue
        txt_name = ".".join(txt_name.split(".")[:-1])
        try:
            imgs_path = glob(f"{imgs_dir}/{txt_name}*")[0]
            ori_img_path = [f for f in glob(f"{txt_dir}/{txt_name}*") if not f.endswith("txt")][0]
        except:
            print(glob(f"{imgs_dir}/{txt_name}*"))
            continue
        img_name = os.path.basename(imgs_path)
        save_path = os.path.join(visualize_path, img_name)
        im = Image.open(imgs_path)
        ori_im = Image.open(ori_img_path)
        w = im.width  # 图片的宽
        h = im.height
        label_info = get_txt_info(txt_file)
        if len(label_info):
            for tmp_info in label_info:
                l_index = int(tmp_info[0])
                label = label_list[l_index]
                l_bbox = [float(i) for i in tmp_info[1:]]
                l_bbox = yolo2labelme(l_bbox, h, w)
                draw_box_label(ori_im, im, l_bbox, label,save_path=save_path)


