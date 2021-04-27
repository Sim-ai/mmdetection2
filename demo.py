# from mmdet.apis import init_detector, inference_detector
# import mmcv
# import cv2
# import demo1
# import numpy as np

# # Specify the path to model config and checkpoint file
# config_file = 'configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
# #checkpoint_file = 'work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/latest.pth'
# checkpoint_file = 'work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/epoch_500.pth'

# # build the model from a config file and a checkpoint file
# #model = init_detector(config_file, checkpoint_file, device='cpu')
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# img = 'demo/demo.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)
# # visualize the results in a new window
# # model.show_result(img, result)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='res/result.jpg')


# # liang
# import glob
# import os
# file_path = "demo/images/"  # 文件夹路径
# savepath = "demo/res/"
# resultpath = "demo/cut/"
# images_path = glob.glob(os.path.join(file_path + '*.png'))  # 所有图片路径
# for image_path in images_path:
#         img = image_path
#         result = inference_detector(model, img)
        
# #         print(result[1].size)
# #         print(result)
#         if result[8].size == 0:
#             continue

            
#         bbox_=[int(i) for i in result[8][0][0:-1]]
        
        
#         image = cv2.imread(img)
#         new_img = image[bbox_[1]:bbox_[3],bbox_[0]:bbox_[2],:]
#         w = new_img.shape[1]
#         h = new_img.shape[0]
        
#         len_w = w//4
#         new_img_1 = new_img[:,0:len_w,:]
#         new_img_2 = new_img[:,len_w:len_w*2,:]
#         new_img_3 = new_img[:,len_w*2:len_w*3,:]
#         new_img_4 = new_img[:,len_w*3:,:]
    
#         cv2.imwrite(resultpath+"new_img_1.jpg",new_img_1)
#         cv2.imwrite(resultpath+"new_img_2.jpg",new_img_2)
#         cv2.imwrite(resultpath+"new_img_3.jpg",new_img_3)
#         cv2.imwrite(resultpath+"new_img_4.jpg",new_img_4)
        
#         demo1.cut_liang()
        
#         pingjie_path="demo/cut/res/"
#         img_name='new_img_'
#         jzg = cv2.imread(pingjie_path+img_name+'1.jpg')  
#         h=jzg.shape[0]
#         w=jzg.shape[1]
#         for i in range(3):
#             #lgz = io.imread(r'C:\Users\123\Desktop\hanjie\crop\dataset\res\W0003_0006_1.jpg')  # np.ndarray, [h, w, c], 值域(0, 255), RGB
#             lgz = cv2.imread(pingjie_path + img_name + str(i + 2) + '.jpg')
#             w1 = lgz.shape[1]
#             pj1 = np.zeros((h, w+w1, 3))  # 横向拼接
#             pj1[:, :w, :] = jzg.copy()  # 图片jzg在左
#             pj1[:, w:, :] = lgz.copy()  # 图片lgz在右
#             pj1 = np.array(pj1, dtype=np.uint8)  # 将pj1数组元素数据类型的改为"uint8"
#             jzg = pj1
#             w=jzg.shape[1]
            
#         print(image_path.split('/')[-1])
#         cv2.imwrite("demo/cut/res_pingjie/"+image_path.split('/')[-1],pj1)    
        
        
        
#         model.show_result(img, result, out_file=savepath + os.path.basename(image_path))
        
#         hh=pj1.shape[0]
#         ww=pj1.shape[1]
#         pj1_new = pj1[1:hh-1,1:ww-1,:]
#         imageeeeee = cv2.imread("demo/res/"+image_path.split('/')[-1])
#         imageeeeee[bbox_[1]+1:bbox_[1]+1+hh-2, bbox_[0]+1:bbox_[0]+1+ww-2, :]=pj1_new.copy()
#         cv2.imwrite("demo/cut/res_last/"+image_path.split('/')[-1],imageeeeee)  
    

    
    
    
    
    
    
    
#东锅数据检测
#两步骤检测，先检测焊缝，再检测小缺陷
# from mmdet.apis import init_detector, inference_detector
# import mmcv
# import cv2
# import demo1
# import numpy as np


# config_file = 'configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
# checkpoint_file = 'work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/Kok/16bit_pth_hanfeng_hist/epoch_10.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

# import glob
# import os

# file_path = "res/"  # 文件夹路径
# savepath = "res/res/"
# resultpath = "res/cut/"
# images_path = glob.glob(os.path.join(file_path + '*.jpg'))  # 所有图片路径

# for image_path in images_path:
#     img = image_path
#     result = inference_detector(model, img)

#     if result[8].size == 0:
#         continue

#     bbox_ = [int(i) for i in result[8][0][0:-1]]

#     image = cv2.imread(img)
#     new_img = image[bbox_[1]:bbox_[3], bbox_[0]:bbox_[2], :]
#     w = new_img.shape[1]
#     h = new_img.shape[0]
#     cv2.imwrite("res/cut/"+image_path.split('/')[-1],new_img)
#     model.show_result(img, result, out_file="res/hanfeng/" + os.path.basename(image_path))

#     demo1.cut_liang()

#     cut_img = cv2.imread("res/res/"+image_path.split('/')[-1])
#     hh = cut_img.shape[0]
#     ww = cut_img.shape[1]
#     pj1_new = cut_img[1:hh - 1, 1:ww - 1, :]
#     imageeeeee = cv2.imread("res/hanfeng/" + image_path.split('/')[-1])
#     imageeeeee[bbox_[1] + 1:bbox_[1] + 1 + hh - 2, bbox_[0] + 1:bbox_[0] + 1 + ww - 2, :] = pj1_new.copy()
#     cv2.imwrite("res/last/" + image_path.split('/')[-1], imageeeeee)


   




















    
    
    
    
 
 

# # 东锅数据集，切成4块检测
# from mmdet.apis import init_detector, inference_detector
# import mmcv
# import cv2
# import demo1
# import numpy as np
# from PIL import Image

# config_file = 'configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
# checkpoint_file = 'work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/Kok/16bit_pth_hanfeng/epoch_100.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # liang
# import glob
# import os
# file_path = "to_xml_test/images/"  # 文件夹路径
# savepath = "demo/res/"
# resultpath = "demo/cut/"
# images_path = glob.glob(os.path.join(file_path + '*.jpg'))  # 所有图片路径
# for image_path in images_path:
#         img = image_path
#         result = inference_detector(model, img)

# #         print(result[1].size)
#         print(result)
#         if result[8].size == 0:
#             continue


#         bbox_=[int(i) for i in result[8][0][0:-1]]


#         # image = cv2.imread(img,2)
#         image = Image.open(img)
#         new_img = image.crop((bbox_[1],bbox_[0],bbox_[3],bbox_[2]))
#         new_img = np.array(new_img)
    
#         w = new_img.shape[1]
#         h = new_img.shape[0]

#         len_w = w//2
#         len_h = h//2
#         new_img_1 = new_img[0:len_h,0:len_w]
#         new_img_2 = new_img[0:len_h,len_w:]
#         new_img_3 = new_img[len_h:,0:len_w]
#         new_img_4 = new_img[len_h:,len_w:]

#         # cv2.imwrite(resultpath+"new_img_1.jpg",new_img_1)
#         # cv2.imwrite(resultpath+"new_img_2.jpg",new_img_2)
#         # cv2.imwrite(resultpath+"new_img_3.jpg",new_img_3)
#         # cv2.imwrite(resultpath+"new_img_4.jpg",new_img_4)
#         new_img_1 = Image.fromarray(new_img_1)
#         new_img_2 = Image.fromarray(new_img_2)
#         new_img_3 = Image.fromarray(new_img_3)
#         new_img_4 = Image.fromarray(new_img_4)
#         new_img_1.save(resultpath+"new_img_1.tif")
#         new_img_2.save(resultpath + "new_img_2.tif")
#         new_img_3.save(resultpath + "new_img_3.tif")
#         new_img_4.save(resultpath + "new_img_4.tif")

#         demo1.cut_liang()

#         pingjie_path="demo/cut/res/"
#         img_name='new_img_'

# #         jzg = cv2.imread(pingjie_path + img_name + '1.jpg')
#         jzg = Image.open(pingjie_path + img_name + '1.tif')
#         jzgg = np.array(jzg)
#         h = jzgg.shape[0]
#         w = jzgg.shape[1]
# #         lgz = cv2.imread(pingjie_path + img_name + str(2) + '.jpg',2)
#         lgz = Image.open(pingjie_path + img_name + '2.tif')
#         pj1 = np.zeros((h, w + w, 3))  # 横向拼接
#         pj1[:, :w] = jzg.copy()  # 图片jzg在左
#         pj1[:, w:] = lgz.copy()  # 图片lgz在右
# #         pj1 = np.array(pj1, dtype=np.uint8)  # 将pj1数组元素数据类型的改为"uint8"

# #         jzg = cv2.imread(pingjie_path + img_name + '3.jpg')  # np.ndarray, [h, w, c], 值域(0, 255), RGB
#         jzg = Image.open(pingjie_path + img_name + '3.tif')
 
# #         lgz = cv2.imread(pingjie_path + img_name + str(4) + '.jpg',2)
#         lgz = Image.open(pingjie_path + img_name + '4.tif')
#         pj2 = np.zeros((h, w + w, 3))  # 横向拼接
#         pj2[:, :w] = jzg.copy()  # 图片jzg在左
#         pj2[:, w:] = lgz.copy()  # 图片lgz在右
# #         pj2 = np.array(pj2, dtype=np.uint8)  # 将pj1数组元素数据类型的改为"uint8"


#         pp1 = np.zeros((h + h, w + w, 3))  # 向拼接
#         pp1[:h, :] = pj1.copy()  # 图片jzg在上
#         pp1[h:, :] = pj2.copy()  # 图片lgz在下
#         # cv2.imwrite("demo/cut/res_pingjie/"+image_path.split('/')[-1],pp1)
#         pp1.save("demo/cut/res_pingjie/"+image_path.split('/')[-1].split('.')[0]+'.tif')


#         model.show_result(img, result, out_file=savepath + os.path.basename(image_path))

#         hh=pp1.shape[0]
#         ww=pp1.shape[1]
#         pp1_new = pp1[1:hh-1,1:ww-1,:]
#         imageeeeee = cv2.imread("demo/res/"+image_path.split('/')[-1])
#         imageeeeee[bbox_[1]+1:bbox_[1]+1+hh-2, bbox_[0]+1:bbox_[0]+1+ww-2, :]=pp1_new.copy()
#         cv2.imwrite("demo/cut/res_last/"+image_path.split('/')[-1],imageeeeee)
        
        
        
        
        
        
# 保存成xml        
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import numpy as np
from PIL import Image        
import glob
import os
from xml.dom.minidom import Document

images_path = 'to_xml_test/images/'
hanfeng_path = 'to_xml_test/hanfeng/'

config_file = 'configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
checkpoint_file ='work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/Kok/16bit_pth_hanfeng_hist/epoch_50.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
checkpoint_file2 = 'work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/Kok/16bit_pth_three_crop_hist/epoch_40.pth'
model2 = init_detector(config_file, checkpoint_file2, device='cuda:0')

for image_path in glob.glob(os.path.join(images_path + '*.jpg')):
    result_hanfeng = inference_detector(model, image_path)
    if result_hanfeng[8].size == 0:
        continue
    bbox_ = [int(i) for i in result_hanfeng[8][0][0:-1]]

    image = cv2.imread(image_path, 2)
    print(image.shape)
    new_img = image[bbox_[1]:bbox_[3], bbox_[0]:bbox_[2]]     #(ymin:ymax,xmin:xmax)
    new_img = Image.fromarray(np.uint16(new_img))
    save_path = os.path.join(hanfeng_path + os.path.basename(image_path).split('.')[0] + '.tif')
    new_img.save(save_path)
    portion = os.path.splitext(save_path)
    if portion[1] ==".tif":
        newname = portion[0]+".jpg"
        os.rename(os.path.join(save_path),os.path.join(newname))
    
    other_path = newname
    result_others = inference_detector(model2, other_path)
    
    doc = Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)
    filename = doc.createElement("filename")
    annotation.appendChild(filename)
    filename.appendChild(doc.createTextNode(image_path))
    
    if result_others[0].size != 0:
        for i in range(0,len(result_others[0])):
            bbox_others = [int(j) for j in result_others[0][i][0:-1]]
            object_name = "气孔"
            xmin_data = bbox_others[0] + bbox_[0]
            ymin_data = bbox_others[1] + bbox_[1]
            xmax_data = bbox_others[2] + bbox_[0]
            ymax_data = bbox_others[3] + bbox_[1]
            object = doc.createElement("object")
            annotation.appendChild(object)
            name = doc.createElement("name")
            object.appendChild(name)
            name.appendChild(doc.createTextNode(object_name))
            bndbox = doc.createElement("bndbox")
            object.appendChild(bndbox)
            xmin = doc.createElement("xmin")
            bndbox.appendChild(xmin)
            xmin.appendChild(doc.createTextNode(str(xmin_data)))
            ymin = doc.createElement("ymin")
            bndbox.appendChild(ymin)
            ymin.appendChild(doc.createTextNode(str(ymin_data)))
            xmax = doc.createElement("xmax")
            bndbox.appendChild(xmax)
            xmax.appendChild(doc.createTextNode(str(xmax_data)))
            ymax = doc.createElement("ymax")
            bndbox.appendChild(ymax)
            ymax.appendChild(doc.createTextNode(str(ymax_data)))
    if result_others[5].size != 0:
        for i in range(0,len(result_others[5])):
            bbox_others = [int(j) for j in result_others[5][i][0:-1]]
            object_name = "未熔合"
            xmin_data = bbox_others[0] + bbox_[0]
            ymin_data = bbox_others[1] + bbox_[1]
            xmax_data = bbox_others[2] + bbox_[0]
            ymax_data = bbox_others[3] + bbox_[1]
            object = doc.createElement("object")
            annotation.appendChild(object)
            name = doc.createElement("name")
            object.appendChild(name)
            name.appendChild(doc.createTextNode(object_name))
            bndbox = doc.createElement("bndbox")
            object.appendChild(bndbox)
            xmin = doc.createElement("xmin")
            bndbox.appendChild(xmin)
            xmin.appendChild(doc.createTextNode(str(xmin_data)))
            ymin = doc.createElement("ymin")
            bndbox.appendChild(ymin)
            ymin.appendChild(doc.createTextNode(str(ymin_data)))
            xmax = doc.createElement("xmax")
            bndbox.appendChild(xmax)
            xmax.appendChild(doc.createTextNode(str(xmax_data)))
            ymax = doc.createElement("ymax")
            bndbox.appendChild(ymax)
            ymax.appendChild(doc.createTextNode(str(ymax_data)))
    if result_others[6].size != 0:
        for i in range(0,len(result_others[6])):
            bbox_others = [int(j) for j in result_others[6][i][0:-1]]
            object_name = "未焊透"
            xmin_data = bbox_others[0] + bbox_[0]
            ymin_data = bbox_others[1] + bbox_[1]
            xmax_data = bbox_others[2] + bbox_[0]
            ymax_data = bbox_others[3] + bbox_[1]
            object = doc.createElement("object")
            annotation.appendChild(object)
            name = doc.createElement("name")
            object.appendChild(name)
            name.appendChild(doc.createTextNode(object_name))
            bndbox = doc.createElement("bndbox")
            object.appendChild(bndbox)
            xmin = doc.createElement("xmin")
            bndbox.appendChild(xmin)
            xmin.appendChild(doc.createTextNode(str(xmin_data)))
            ymin = doc.createElement("ymin")
            bndbox.appendChild(ymin)
            ymin.appendChild(doc.createTextNode(str(ymin_data)))
            xmax = doc.createElement("xmax")
            bndbox.appendChild(xmax)
            xmax.appendChild(doc.createTextNode(str(xmax_data)))
            ymax = doc.createElement("ymax")
            bndbox.appendChild(ymax)
            ymax.appendChild(doc.createTextNode(str(ymax_data)))
    
    filename_save = image_path.split('.')[0]+".xml"
    f = open(filename_save, "w",encoding="utf8")
    f.write(doc.toprettyxml(indent="  "))
    f.close()
        
        
        
        
        
        
