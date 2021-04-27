from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2

# Specify the path to model config and checkpoint file
config_file = 'configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
# checkpoint_file = 'work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/latest.pth'
checkpoint_file = 'work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/Kok/16bit_pth_three_crop_hist/epoch_40.pth'

# build the model from a config file and a checkpoint file
#model = init_detector(config_file, checkpoint_file, device='cpu')
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# liang
import glob
import os

def cut_liang():
    file_path = "to_xml_test/hanfeng/"  # 文件夹路径
    savepath = "to_xml_test/"
    images_path = glob.glob(os.path.join(file_path + '*.jpg'))  # 所有图片路径
    for image_path in images_path:
        img = image_path
        result = inference_detector(model, img)
        model.show_result(img, result, out_file=savepath + os.path.basename(image_path))

