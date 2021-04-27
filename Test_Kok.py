from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import tqdm
#config_file = 'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
#checkpoint_file = 'work_dirs/cascade_rcnn_r50_fpn_1x_coco/epoch_10.pth'

config_file = 'configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
# checkpoint_file = 'work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/latest.pth'
checkpoint_file = 'work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/Kok/16bit_pth_three_crop_hist/epoch_40.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

import glob
import os
# 生成csv文件
import csv
# 表头
def annotations_information(information = []):
    field_order = ["name", 'image_id', 'confidence','xmin','ymin','xmax','ymax']
    with open("Kok_data/res/myresult.csv", 'w', encoding="utf-8", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, field_order)
        writer.writeheader()
        for inf in information:       
            #writer.writerow(dict(zip(field_order, ["张三", 20, "男"])))
            writer.writerow(dict(zip(field_order, inf)))


annotations_list = []
# 个人数据
file_path = "Kok_data/"  # 文件夹路径
savepath = "Kok_data/res/"

images_path = glob.glob(os.path.join(file_path + '*.jpg'))  # 所有图片路径
images_path.sort()
for image_path in tqdm.tqdm(images_path):
    img = image_path
    result = inference_detector(model, img)
    model.show_result(img, result, out_file=savepath + os.path.basename(image_path))
    for j,category in enumerate(result): # 遍历每一类
        if j == 0:
            object_type = 'holothurian'
        elif j == 1:
            object_type = 'echinus'
        elif j == 2:
            object_type = 'scallop'
        elif j == 3:
            object_type = 'starfish'
        for res in category:
            [xmin,ymin,xmax,ymax,p] = res
            annotation = [str(object_type),str(img[9:-4]),float(p),int(xmin),int(ymin),int(xmax),int(ymax)]
            annotations_list.append(annotation)
annotations_information(annotations_list)
# voc数据
'''
txt_file_path = "./data/VOCdevkit/VOC2007/ImageSets/Main/test.txt"  # 文件夹路径
img_file_path = "./data/VOCdevkit/VOC2007/JPEGImages"
savepath = "data/VOCdevkit/result/"
img_path_list = []
with open(txt_file_path, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        img_path_list.append(os.path.join(img_file_path,line+'.jpg'))
#print(img_path_list)
for image_path in tqdm.tqdm(img_path_list):
    img = image_path
    result = inference_detector(model, img)
    model.show_result(img, result, out_file=savepath + os.path.basename(image_path))
'''