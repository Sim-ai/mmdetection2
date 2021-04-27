'''
切割图像，同时切割对应的标注文件，使得切割得到的子图像对应的标注文件包含切割后的目标;
目前支持voc和yolo格式的标注文件;
'''

import os
import argparse
import glob
from PIL import Image
from itertools import product
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import copy

try:
	import xml.etree.cElementTree as ET
except ImportError:
	import xml.etree.ElementTree as ET
np.set_printoptions(suppress=True)

suffix = ["tif", "tiff", "png", "jpg", "jpeg", "bmp"]


def prettyXml(element, indent, newline, level = 0):
	if element:
		if element.text == None or element.text.isspace():
			element.text = newline + indent * (level + 1)
		else:
			element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
	temp = list(element)
	for subelement in temp:
		if temp.index(subelement) < (len(temp) - 1):
			subelement.tail = newline + indent * (level + 1)
		else:
			subelement.tail = newline + indent * level
		prettyXml(subelement, indent, newline, level = level + 1)


def crop_yolo(img_path, format="tif", h_slice=1, w_slice=1):
	# Deprecated
	# img_suffix = os.path.splitext(img_path)[-1]
	label_path = os.path.join(os.path.dirname(img_path), os.path.splitext(os.path.basename(img_path))[0] + ".txt")
	if not os.path.exists(label_path):
		return

	img = ToTensor()(Image.open(img_path))

	_, h, w = img.shape
	anno = np.loadtxt(label_path)

	if len(anno.shape) != 2:
		anno = anno[None]
	anno[:, 1] *= w
	anno[:, 2] *= h
	anno[:, 3] *= w
	anno[:, 4] *= h

	anno[:, 1] = anno[:, 1] - anno[:, 3] / 2
	anno[:, 2] = anno[:, 2] - anno[:, 4] / 2
	anno[:, 3] = anno[:, 1] + anno[:, 3]
	anno[:, 4] = anno[:, 2] + anno[:, 4]

	_h = h + 1 - h % h_slice
	_w = w + 1 - w % w_slice
	h_grid = np.arange(0, _h, h // h_slice)
	w_grid = np.arange(0, _w, w // w_slice)

	lt = []
	rb = []
	coor_slice = list(product(h_grid, w_grid))

	for r in range(h_slice):
		for i in range(r * (w_slice + 1), r * (w_slice + 1) + w_slice):
			lt.append(coor_slice[i])
			rb.append(coor_slice[i + w_slice + 2])

	for _i, (_lt, _rb) in enumerate(list(zip(lt, rb))):
		sub_lty, sub_ltx = _lt
		sub_rby, sub_rbx = _rb
		sub_height, sub_width = sub_rby - sub_lty, sub_rbx - sub_ltx

		_label = anno[:, 0]
		_ltx = np.clip(anno[:, 1], sub_ltx, sub_rbx)
		_lty = np.clip(anno[:, 2], sub_lty, sub_rby)
		_rbx = np.clip(anno[:, 3], sub_ltx, sub_rbx)
		_rby = np.clip(anno[:, 4], sub_lty, sub_rby)
		clipped_coor = np.stack([_label, _ltx, _lty, _rbx, _rby], 1)
		_idx = np.copy(clipped_coor[:, 1:])
		new_label = np.copy(clipped_coor)
		_idx[:, 2] -= _idx[:, 0]
		_idx[:, 3] -= _idx[:, 1]

		# new_cx
		new_label[:, 1] = ((clipped_coor[:, 1] + clipped_coor[:, 3]) / 2 - sub_ltx) / sub_width
		# new_cy
		new_label[:, 2] = ((clipped_coor[:, 2] + clipped_coor[:, 4]) / 2 - sub_lty) / sub_height
		# new_w
		new_label[:, 3] = (clipped_coor[:, 3] - clipped_coor[:, 1]) / sub_width
		# new_h
		new_label[:, 4] = (clipped_coor[:, 4] - clipped_coor[:, 2]) / sub_height

		if not os.path.exists(os.path.join(os.path.dirname(img_path), "yolo")):
			os.makedirs(os.path.join(os.path.dirname(img_path), "yolo"))

		sub_img_path = os.path.join(os.path.join(os.path.dirname(img_path), "yolo"), f"{os.path.splitext(os.path.basename(img_path))[0]}_{_i}.{format}")
		sub_label_path = os.path.join(os.path.join(os.path.dirname(img_path), "yolo"), f"{os.path.splitext(os.path.basename(img_path))[0]}_{_i}.txt")

		sub_label = new_label[~np.any(_idx == 0, axis=1)]
		if not sub_label.size:
			continue
		sub_img = ToPILImage()(img[:, sub_lty:sub_rby, sub_ltx:sub_rbx])
		sub_img.save(sub_img_path)

		np.savetxt(sub_label_path, new_label[~np.any(_idx[:, 2:] < 1e-4, axis=1) & ~np.any(new_label[:, 3:] < 1e-4, axis=1)], fmt="%.05f")


def crop_voc(image_path, format="tif", h_slice=1, w_slice=1):
	label_path = os.path.join(os.path.dirname(image_path), os.path.splitext(os.path.basename(image_path))[0] + ".xml")
	if not os.path.exists(label_path):
		return
	# Deprecated
	# image_suffix = os.path.splitext(image_path)[-1]

	if not os.path.exists(os.path.join(os.path.dirname(image_path), "voc")):
		os.makedirs(os.path.join(os.path.dirname(image_path), "voc"))
	print(Image.open(image_path))
	image = ToTensor()(Image.open(image_path))
	_, h, w = image.shape
	tree = ET.ElementTree(file=label_path)
	root = tree.getroot()
	ObjectSet = root.findall("object")

	_h = h + 1 - h % h_slice
	_w = w + 1 - w % w_slice
	h_grid = np.arange(0, _h, h // h_slice)
	w_grid = np.arange(0, _w, w // w_slice)

	lt = []
	rb = []
	coor_slice = list(product(h_grid, w_grid))

	for r in range(h_slice):
		for i in range(r * (w_slice + 1), r * (w_slice + 1) + w_slice):
			lt.append(coor_slice[i])
			rb.append(coor_slice[i + w_slice + 2])

	divided_objects = {}
	for _i, (_lt, _rb) in enumerate(list(zip(lt, rb))):
		sub_lty, sub_ltx = _lt
		sub_rby, sub_rbx = _rb
		sub_height, sub_width = sub_rby - sub_lty, sub_rbx - sub_ltx

		sub_image_path = os.path.join(os.path.join(os.path.dirname(image_path), "voc"),
									  f"{os.path.splitext(os.path.basename(image_path))[0]}_{_i}.{format}")
		sub_anno_path = os.path.join(os.path.join(os.path.dirname(image_path), "voc"),
									 f"{os.path.splitext(os.path.basename(image_path))[0]}_{_i}.xml")
		divided_objects[sub_image_path] = []

		for object in ObjectSet:
			obj_box = object.find("bndbox")
			x1 = int(obj_box.find("xmin").text)
			y1 = int(obj_box.find("ymin").text)
			x2 = int(obj_box.find("xmax").text)
			y2 = int(obj_box.find("ymax").text)
			_box = np.array([x1, y1, x2, y2])

			sub_box = np.copy(_box)

			## make sure the sub_box is within [1, sub_size - 1]
			sub_box[0] = np.clip(sub_box[0], sub_ltx + 1, sub_rbx - 1)
			sub_box[1] = np.clip(sub_box[1], sub_lty + 1, sub_rby - 1)
			sub_box[2] = np.clip(sub_box[2], sub_ltx + 1, sub_rbx - 1)
			sub_box[3] = np.clip(sub_box[3], sub_lty + 1, sub_rby - 1)
			delta_box = np.copy(sub_box)

			delta_box[2] -= delta_box[0]
			delta_box[3] -= delta_box[1]

			if ~np.any(delta_box[2:] < 1):
				sub_object = ET.Element("object")
				sub_object.append(object.find("name"))
				sub_object.append(object.find("pose"))
				sub_object.append(object.find("truncated"))
				sub_object.append(object.find("difficult"))

				sub_bndbox = ET.Element("bndbox")
				bndbox_xmin = ET.Element("xmin")
				bndbox_xmin.text = str(int(sub_box[0]) - sub_ltx)
				sub_bndbox.append(bndbox_xmin)
				bndbox_ymin = ET.Element("ymin")
				bndbox_ymin.text = str(int(sub_box[1]) - sub_lty)
				sub_bndbox.append(bndbox_ymin)
				bndbox_xmax = ET.Element("xmax")
				bndbox_xmax.text = str(int(sub_box[2]) - sub_ltx)
				sub_bndbox.append(bndbox_xmax)
				bndbox_ymax = ET.Element("ymax")
				bndbox_ymax.text = str(int(sub_box[3]) - sub_lty)
				sub_bndbox.append(bndbox_ymax)
				sub_object.append(sub_bndbox)
				divided_objects[sub_image_path].append(sub_object)

		if len(divided_objects[sub_image_path]) != 0:
			sub_image = ToPILImage()(image[:, sub_lty:sub_rby, sub_ltx:sub_rbx])

			sub_xml = copy.deepcopy(tree)
			sub_root = sub_xml.getroot()
			sub_folder = sub_root.find("folder")
			if sub_folder is not None:
				sub_folder.text = os.path.split(os.path.split(sub_image_path)[0])[1]
			sub_filename = sub_root.find("filename")
			if sub_filename is not None:
				sub_filename.text = os.path.basename(sub_image_path)
			sub_path = sub_root.find("path")
			if sub_path is not None:
				sub_path.text = os.path.abspath(sub_image_path)
			sub_size = sub_root.find("size")
			sub_size.find("width").text = str(int(sub_width))
			sub_size.find("height").text = str(int(sub_height))

			object_set = sub_root.findall("object")
			for object in object_set:
				sub_root.remove(object)
			for sub_object in divided_objects[sub_image_path]:
				sub_root.append(sub_object)

			sub_image.save(sub_image_path)
			prettyXml(sub_root, "\t", os.linesep)
			sub_xml.write(sub_anno_path, encoding="utf-8")


def pre_process(root_dir, mode="voc", format="tif", h_slice=1, w_slice=1):
	image_list = sum([glob.glob(f"{root_dir}/*.{_suffix}") for _suffix in suffix], [])
	for image_path in image_list:
		if mode == "yolo":
			crop_yolo(image_path, format=format, h_slice=h_slice, w_slice=w_slice)
		elif mode == "voc":
			crop_voc(image_path, format=format, h_slice=h_slice, w_slice=w_slice)
		else:
			raise ValueError("Invalid mode, please check your input")


def crop_only(root_dir, format="tif", h_slice=1, w_slice=1):
	image_list = sum([glob.glob(f"{root_dir}/*.{_suffix}") for _suffix in suffix], [])

	if not os.path.exists(os.path.join(root_dir, "cropped_images")):
		os.makedirs(os.path.join(root_dir, "cropped_images"))

	for image_path in image_list:
		# Deprecated
		# image_suffix = os.path.splitext(image_path)[-1]
		image = ToTensor()(Image.open(image_path))
		_, h, w = image.shape

		_h = h + 1 - h % h_slice
		_w = w + 1 - w % w_slice
		h_grid = np.arange(0, _h, h // h_slice)
		w_grid = np.arange(0, _w, w // w_slice)

		lt = []
		rb = []
		coor_slice = list(product(h_grid, w_grid))

		for r in range(h_slice):
			for i in range(r * (w_slice + 1), r * (w_slice + 1) + w_slice):
				lt.append(coor_slice[i])
				rb.append(coor_slice[i + w_slice + 2])

		for _i, (_lt, _rb) in enumerate(list(zip(lt, rb))):
			sub_lty, sub_ltx = _lt
			sub_rby, sub_rbx = _rb

			sub_image_path = os.path.join(os.path.join(root_dir, "cropped_images"),
										  f"{os.path.splitext(os.path.basename(image_path))[0]}_{_i}.{format}")
			sub_image = ToPILImage()(image[:, sub_lty:sub_rby, sub_ltx:sub_rbx])
			sub_image.save(sub_image_path)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="dataset", help="the directory containing all images and annotations")
	parser.add_argument("--mode", type=str, default="voc", help="the annotation protocol u use")
	parser.add_argument("--format", type=str, default="tif", help="the format of sub images u want")
	parser.add_argument("--h_slice", type=int, default=1, help="how many rows u want")
	parser.add_argument("--w_slice", type=int, default=1, help="how many cols u want")
	parser.add_argument("--crop_only", action="store_true", help="crop images only")

	opt = parser.parse_args()
	root_dir = os.path.abspath(opt.data_dir)

	if opt.crop_only:
		crop_only(root_dir, format=opt.format, h_slice=opt.h_slice, w_slice=opt.w_slice)
	else:
		pre_process(root_dir, mode=opt.mode, format=opt.format, h_slice=opt.h_slice, w_slice=opt.w_slice)