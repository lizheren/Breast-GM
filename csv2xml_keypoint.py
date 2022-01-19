import os
import numpy as np
import codecs
import pandas as pd
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split
from IPython import embed

# 1.标签路径

csv_file = "./../csv_with_feats_20211124/train.csv"
# # csv_file = "./data/Medical_image/train.csv"
# saved_path = "./VOCdevkit/annotations/"  # 保存路径
# image_save_path = "./JPEGImages/"
# image_raw_parh = "../csv/images/"
# # 2.创建要求文件夹
# if not os.path.exists(saved_path + "Annotations"):
#     os.makedirs(saved_path + "Annotations")
# if not os.path.exists(saved_path + "JPEGImages/"):
#     os.makedirs(saved_path + "JPEGImages/")
# if not os.path.exists(saved_path + "ImageSets/Main/"):
#     os.makedirs(saved_path + "ImageSets/Main/")



saved_path = "./annotations/"  # 保存路径
if not os.path.exists(saved_path + "breast"):
    os.makedirs(saved_path + "breast")




# 3.获取待处理文件
total_csv_annotations = {}
annotations = pd.read_csv(csv_file)
# for annotation in annotations:
#     key = annotation[0].split(os.sep)[-1]
#     value = np.array([annotation[1:]])
#     print(key)
#     print(value)
#     if key in total_csv_annotations.keys():
#         total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
#     else:
#         total_csv_annotations[key] = value
for i in range(len(annotations)):
    key = annotations.loc[i, 'SOPInstanceUID']
    total_csv_annotations[key] = np.array([[annotations.loc[i, 'StartX'], annotations.loc[i, 'StartY'],
                                            annotations.loc[i, 'EndX'], annotations.loc[i, 'EndX'], 'mass']])
    total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],
                                                 np.array([[annotations.loc[i, 'nipple_StartX'],
                                                            annotations.loc[i, 'nipple_StartY'],
                                                            annotations.loc[i, 'nipple_EndX'],
                                                            annotations.loc[i, 'nipple_EndY'], 'nipple']]))
                                                , axis=0)
    total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],
                                                 np.array([[annotations.loc[i, 'lesion2Chest_StartX'],
                                                            annotations.loc[i, 'lesion2Chest_StartY'],
                                                            annotations.loc[i, 'lesion2Chest_EndX'],
                                                            annotations.loc[i, 'lesion2Chest_EndY'], 'chest']]))
                                                , axis=0)
    total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],
                                                 np.array([[annotations.loc[i, 'lesion2Skin_StartX'],
                                                            annotations.loc[i, 'lesion2Skin_StartY'],
                                                            annotations.loc[i, 'lesion2Skin_EndX'],
                                                            annotations.loc[i, 'lesion2Skin_EndY'], 'skin']]))
                                                , axis=0)




for filename, label in total_csv_annotations.items():
    # embed()
    # height, width, channels = 256, 256, 3  # cv2.imread(image_raw_parh + filename).shape
    # embed()
    with codecs.open(saved_path + "breast/" + filename + "_1.xml", "w", "utf-8") as xml:
        xml.write('<?xml version="1.0" encoding="utf-8"?>\n')
        xml.write('<annotation>\n')
        xml.write('\t<image>' + filename + '</image>\n')
        xml.write('\t<voc_id>1</voc_id>\n')
        xml.write('\t<category>breast</category>\n')
        xml.write('\t<visible_bounds height="256.00" width="256.00" xmin="0.00" ymin="256.00"/>\n')
        xml.write('\t<keypoints>\n')


        for label_detail in label:
            labels = label_detail
            # embed()
            xmin = int(labels[0])
            ymin = int(labels[1])
            xmax = int(labels[2])
            ymax = int(labels[3])
            label_ = labels[-1]
            # if xmax <= xmin:
            #     temp = xmax
            #     xmax = xmin
            #     xmin = temp
            # elif ymax <= ymin:
            #     tempy = ymax
            #     ymax = ymin
            #     ymin = tempy
            # else:
            keypoint_x = round((xmax + xmin)/2, 2)
            keypoint_y = round((ymax + ymin) / 2, 2)
            xml.write('\t\t<keypoint name=\"' + label_ + '\" visible="1" x=\"'+ str(keypoint_x) + '\" y=\"'+ str(keypoint_y) + '\" z="0.00"/>\n')
            print(label_, keypoint_x, keypoint_y)
        xml.write('\t</keypoints>\n')
        xml.write('</annotation>')




    # key = ''
    # value_1 = 'StartX'
    # value_2 = ''
# total_csv_annotations
# 4.读取标注信息并写入 xml
# for filename, label in total_csv_annotations.items():
#     # embed()
#     height, width, channels = 256, 256, 3  # cv2.imread(image_raw_parh + filename).shape
#     # embed()
#     with codecs.open(saved_path + "Annotations/" + filename + ".xml", "w", "utf-8") as xml:
#         xml.write('<annotation>\n')
#         xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
#         xml.write('\t<filename>' + filename + '</filename>\n')
#         xml.write('\t<source>\n')
#         xml.write('\t\t<database>The UAV autolanding</database>\n')
#         xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
#         xml.write('\t\t<image>flickr</image>\n')
#         xml.write('\t\t<flickrid>NULL</flickrid>\n')
#         xml.write('\t</source>\n')
#         # xml.write('\t<owner>\n')
#         # xml.write('\t\t<flickrid>NULL</flickrid>\n')
#         # xml.write('\t\t<name>ChaojieZhu</name>\n')
#         # xml.write('\t</owner>\n')
#         xml.write('\t<size>\n')
#         xml.write('\t\t<width>' + str(width) + '</width>\n')
#         xml.write('\t\t<height>' + str(height) + '</height>\n')
#         xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
#         xml.write('\t</size>\n')
#         xml.write('\t\t<segmented>0</segmented>\n')
#         if isinstance(label, float):
#             ## 空白
#             xml.write('</annotation>')
#             continue
#         for label_detail in label:
#             labels = label_detail
#             # embed()
#             xmin = int(labels[0])
#             ymin = int(labels[1])
#             xmax = int(labels[2])
#             ymax = int(labels[3])
#             label_ = labels[-1]
#             if xmax <= xmin:
#                 temp = xmax
#                 xmax = xmin
#                 xmin = temp
#             elif ymax <= ymin:
#                 tempy = ymax
#                 ymax = ymin
#                 ymin = tempy
#             else:
#                 xml.write('\t<object>\n')
#                 xml.write('\t\t<name>' + label_ + '</name>\n')
#                 xml.write('\t\t<pose>Unspecified</pose>\n')
#                 xml.write('\t\t<truncated>1</truncated>\n')
#                 xml.write('\t\t<difficult>0</difficult>\n')
#                 xml.write('\t\t<bndbox>\n')
#                 xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
#                 xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
#                 xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
#                 xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
#                 xml.write('\t\t</bndbox>\n')
#                 xml.write('\t</object>\n')
#                 print(filename, xmin, ymin, xmax, ymax, label_)
#         xml.write('</annotation>')

# 6.split files for txt
# txtsavepath = saved_path + "ImageSets/Main/"
# ftrainval = open(txtsavepath+'/trainval.txt', 'w')
# ftest = open(txtsavepath+'/test.txt', 'w')
# ftrain = open(txtsavepath+'/train.txt', 'w')
# fval = open(txtsavepath+'/val.txt', 'w')
# total_files = glob(saved_path+"./Annotations/*.xml")
# total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
# #test_filepath = ""
# for file in total_files:
#     ftrainval.write(file + "\n")
#
#
#
# train_files,val_files = train_test_split(total_files,test_size=0.15,random_state=42)
#
# for file in train_files:
#     ftrain.write(file + "\n")
# #val
# for file in val_files:
#     fval.write(file + "\n")
#
# ftrainval.close()
# ftrain.close()
# fval.close()
# #ftest.close()