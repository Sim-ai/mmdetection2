#直方图均衡化
from PIL import Image
import tqdm
import numpy as np
import cv2
import os
root_dir = r"./Kok_data/" # 数据路径
'''
for filename in tqdm.tqdm(os.listdir(root_dir)):
    if os.path.splitext(filename)[1] == '.jpg':
        img = Image.open(os.path.join(root_dir, filename))
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=7, tileGridSize=(6, 6))
        dst = clahe.apply(img)
        dst = cv2.medianBlur(dst, 5)
        dst = Image.fromarray(np.uint16(dst))
        dst.save(os.path.join(root_dir, filename[:-3]+'tif'))
        os.remove(os.path.join(root_dir, filename))
'''
files = os.listdir(root_dir)
for filename in files:
    portion = os.path.splitext(filename)#分离文件名字和后缀
    #    print(portion)

    if portion[1] ==".tif":#根据后缀来修改,如无后缀则空
        newname = portion[0]+".jpg"#要改的新后缀
            #os.chdir("D:/ruanjianxiazai/tuxiangyangben/fengehou/train")#切换文件路径,如无路径则要新建或者路径同上,做好备份
        os.rename(os.path.join(root_dir,filename),os.path.join(root_dir,newname))
