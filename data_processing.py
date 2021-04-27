import os
import shutil
import glob
from PIL import Image
import argparse
import os
import tqdm
import xml.etree.ElementTree as ET
import random
import numpy as np
import cv2
from crop_image_and_objects import pre_process,crop_only
from mmdet.apis import init_detector, inference_detector
import mmcv

#生成VOC数据
suffix = ["tif", "tiff", "png", "jpg", "jpeg", "bmp"]
def voc_allocator(root_dir, trainval_per=0.8, train_per=0.8):
	os.makedirs(os.path.join(root_dir, "VOCdevkit", "VOC2007", "Annotations"), exist_ok=True)
	os.makedirs(os.path.join(root_dir, "VOCdevkit", "VOC2007", "ImageSets", "Main"), exist_ok=True)
	os.makedirs(os.path.join(root_dir, "VOCdevkit", "VOC2007", "JPEGImages"), exist_ok=True)

	image_root = root_dir
	anno_root = root_dir
	image_dst = os.path.join(root_dir, "VOCdevkit", "VOC2007", "JPEGImages")
	anno_dst = os.path.join(root_dir, "VOCdevkit", "VOC2007", "Annotations")

	image_list = sum([glob.glob(f"{image_root}/*.{_suffix}") for _suffix in suffix], [])
	for image_path in image_list:
		label_path = os.path.join(anno_root, os.path.splitext(os.path.basename(image_path))[0]+".xml")
		if os.path.exists(label_path):
			image = Image.open(image_path)
			image.save(os.path.join(image_dst, os.path.splitext(os.path.basename(image_path))[0]+".tif"))
			shutil.copyfile(label_path, os.path.join(anno_dst, os.path.basename(label_path)))


	os.chdir(os.path.join(root_dir, "VOCdevkit", "VOC2007"))

	xmlfilepath = 'Annotations'
	txtsavepath = os.path.join("ImageSets", "Main")
	total_xml = os.listdir(xmlfilepath)

	num = len(total_xml)
	list = range(num)
	tv = int(num * trainval_per)
	tr = int(tv * train_per)
	trainval = random.sample(list, tv)
	train = random.sample(trainval, tr)

	ftrainval = open(os.path.join(txtsavepath, "trainval.txt"), 'w')
	ftest = open(os.path.join(txtsavepath, "test.txt"), 'w')
	ftrain = open(os.path.join(txtsavepath, "train.txt"), 'w')
	fval = open(os.path.join(txtsavepath, "val.txt"), 'w')

	for i in list:
		name = total_xml[i][:-4] + '\n'
		if i in trainval:
			ftrainval.write(name)
			if i in train:
				ftrain.write(name)
			else:
				fval.write(name)
		else:
			ftest.write(name)

	ftrainval.close()
	ftrain.close()
	fval.close()
	ftest.close()

# 图片裁剪,删除多余图像中空白地方
def delet_blank(root_dir):
	'''
	参数：
	root_dir: 文件路径
	'''
	for filename in tqdm.tqdm(os.listdir(root_dir)):
		if filename[-3:] == 'jpg':
			img_path = os.path.join(root_dir,filename)
			image = Image.open(img_path)
			if image.size == (1200, 850):
				new_image = image.crop((0,0,850,850))
				new_image.save(root_dir + filename[:-3] + 'tif')
				os.remove(img_path)

# 拆分6块的代码
def crop_iamge(root_dir, h_slice, w_slice):
	'''
	参数：
	root_dir: 文件路径
	mode: 数据模式
	formats: 保存数据类型
	h_slice: 高度划分块数
	w_slice: 宽度划分块数
	'''
	mode="voc"
	formats="tif"
	pre_process(root_dir,mode,formats,h_slice,w_slice)
	# 将拆分的数据放在voc文件夹
	# 对xml和img进行拆分
	xml_path = os.path.join(root_dir,'voc')
	for filename in tqdm.tqdm(os.listdir(xml_path)):
		if filename[-3:] == 'xml':
			tree = ET.parse(os.path.join(xml_path, filename))
			root = tree.getroot()
			for object in root.findall('object'):
				if object.find('name').text == '焊缝':
					xmin = int(object.find('bndbox').find('xmin').text)
					ymin = int(object.find('bndbox').find('ymin').text)
					xmax = int(object.find('bndbox').find('xmax').text)
					ymax = int(object.find('bndbox').find('ymax').text)
					if xmax-xmin < 20 or ymax -ymin < 20:
						root.remove(object)
				tree.write(os.path.join(xml_path, filename),encoding='utf-8')
	## 删除多余文件
	filelist = os.listdir(xml_path)
	for filename in os.listdir(xml_path):
		if filename[-3:] == 'tif':
			if filename[:-3]+'xml' not in filelist:
				os.remove(os.path.join(path,filename[:-3]+'tif'))

# 删除除焊缝外的其他类别的缺陷
def delete_annotations(root_dir,list=[]):
	xml_path_list = []
	for filename in os.listdir(root_dir):
		if filename[-3:] == 'xml':
			xml_path_list.append(os.path.join(root_dir, filename))

		for xml in tqdm.tqdm(xml_path_list):
			# 从xml文件中读取，使用getroot()获取根节点，得到的是一个Element对象
			tree = ET.parse(xml)
			root = tree.getroot()

			for object in root.findall('object'):
				deleted = str(object.find('name').text)
				if (deleted not in list):
					print("deleted：", deleted)
					root.remove(object)
			tree.write(xml,encoding='utf-8')

#直方图均衡化
def data_hist(root_dir):
	for filename in tqdm.tqdm(os.listdir(root_dir)):
		if os.path.splitext(filename)[1] == '.tif' or os.path.splitext(filename)[1] == '.jpg':
			img = Image.open(os.path.join(root_dir, filename))
			img = np.array(img)
			clahe = cv2.createCLAHE(clipLimit=7, tileGridSize=(6, 6))
			dst = clahe.apply(img)
			dst = cv2.medianBlur(dst, 5)
			dst = Image.fromarray(np.uint16(dst))
			dst.save(os.path.join(root_dir, filename[:-3]+'tif'))
	tif_to_jpg(root_dir)
# tif文件转jpg文件
def tif_to_jpg(root_dir):
	for filename in os.listdir(root_dir):
		portion = os.path.splitext(filename)#分离文件名字和后缀
		if portion[1] ==".tif":#根据后缀来修改,如无后缀则空
			newname = portion[0]+".jpg"#要改的新后缀
			os.rename(os.path.join(root_dir,filename),os.path.join(root_dir,newname))

# 模型检测
def myTest(root_dir, checkpoint_file_hanfeng, checkpoint_file_others):
	config_file = 'configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
	res_path = os.path.join(root_dir,"res/")
	bbox_all = []
	# 焊缝检测
	checkpoint_file = os.path.join('work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/',checkpoint_file_hanfeng)	
	model = init_detector(config_file, checkpoint_file, device='cuda:0')
	images_path = glob.glob(os.path.join(root_dir + '/*.jpg'))  # 所有图片路径
	for image_path in tqdm.tqdm(images_path):
		result_hanfeng = inference_detector(model, image_path)
		if result_hanfeng[8].size == 0:
			continue
		bbox_ = [int(i) for i in result_hanfeng[8][0][0:-1]]
		bbox_all.append(bbox_)

		image = cv2.imread(image_path, 2)
		new_img = image[bbox_[1]:bbox_[3], bbox_[0]:bbox_[2]]
		new_img = Image.fromarray(np.uint16(new_img))
		save_path = os.path.join(root_dir, "tmp/"+ os.path.basename(image_path).split('.')[0] + '.tif')
		new_img.save(save_path)
		tif_to_jpg(os.path.join(root_dir, "tmp/"))
	
	# 其他缺陷检测
	checkpoint_file = os.path.join('work_dirs/cascade_rcnn_hrnetv2p_w32_20e_coco/',checkpoint_file_others)	
	model = init_detector(config_file, checkpoint_file, device='cuda:0')
	others_path = glob.glob(os.path.join(root_dir+'/tmp' + '/*.jpg'))  # 所有图片路径
	for index,other_path in enumerate(others_path):
		others_result = inference_detector(model, other_path)
		model.show_result(other_path, others_result, out_file=res_path + os.path.basename(other_path))
		cut_img = cv2.imread(res_path + os.path.basename(other_path))
		hh = cut_img.shape[0]
		ww = cut_img.shape[1]
		ori_img = cv2.imread(os.path.join(root_dir, os.path.basename(other_path)))
		ori_img[bbox_all[index][1]:bbox_all[index][1] + hh, bbox_all[index][0]:bbox_all[index][0] + ww ,:] = cut_img.copy()
		cv2.imwrite(res_path + os.path.basename(other_path), ori_img)


    
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-mt", "--model_test", type=bool,default=False, help="**")
	parser.add_argument("-rd", "--root_dir", type=str, default="dataset", help="the directory containing all images and annotations")
	parser.add_argument("-ic", "--img_crop", type=bool, default=False, help="crop the image")
	parser.add_argument("-hs", "--h_slice", type=int, default=1, help="how many rows u want")
	parser.add_argument("-ws", "--w_slice", type=int, default=1, help="how many cols u want")
	parser.add_argument("-al", "--annotation_list", type=str, default="焊缝,气孔,未熔合,未焊透", help="delete annotations")
	parser.add_argument("-gv", "--get_VOCdevkit", type=bool, default=False, help="**")
	parser.add_argument("--trainval_per", type=float, default=0.8, help="the fraction of trainval within the total dataset")
	parser.add_argument("--train_per", type=float, default=0.8, help="the fraction of train within trainval")
	parser.add_argument("-cfh","--checkpoint_file_hanfeng", type=str, default='Kok/16bit_pth_hanfeng_hist/epoch_10.pth', help="**")
	parser.add_argument("-cfo","--checkpoint_file_others", type=str, default='Kok/16bit_pth_three_crop_hist/epoch_20.pth', help="**")
	opt = parser.parse_args()
	root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),opt.root_dir)
	if opt.model_test == False:
		annotation_list = opt.annotation_list.split(',')
		
		if opt.img_crop == True:
			crop_iamge(root_dir, opt.h_slice, opt.w_slice)
			root_dir = os.path.join(root_dir,'voc')
		
		delete_annotations(root_dir, annotation_list)
		
		if opt.get_VOCdevkit == True:
			voc_allocator(root_dir, opt.trainval_per, opt.train_per)
		data_hist(os.path.join(root_dir,'VOCdevkit/VOC2007/JPEGImages'))
	else:
		if opt.img_crop == True:
			crop_only(root_dir, h_slice = opt.h_slice, w_slice =opt.w_slice)
			root_dir = os.path.join(root_dir,'cropped_images/')
		data_hist(os.path.join(root_dir))
		myTest(root_dir,opt.checkpoint_file_hanfeng, opt.checkpoint_file_others)
