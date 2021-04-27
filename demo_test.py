from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2

# Specify the path to model config and checkpoint file
config_file = 'configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
# checkpoint_file = 'work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/latest.pth'
checkpoint_file = 'work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/epoch_120.pth'

# build the model from a config file and a checkpoint file
#model = init_detector(config_file, checkpoint_file, device='cpu')
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo/demo.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='res/result.jpg')

# liang
import glob
import os


file_path = "demo/test_Kok/"  # 文件夹路径
savepath = "demo/test_Kok/res/"
images_path = glob.glob(os.path.join(file_path + '*.jpg'))  # 所有图片路径
for image_path in images_path:
    img = image_path
    result = inference_detector(model, img)
    model.show_result(img, result, out_file=savepath + os.path.basename(image_path))