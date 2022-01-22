from pathlib import Path
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import random
import pickle
import torch
import sys

from utils.config import cfg

anno_path = cfg.Breastdata.KPT_ANNO_DIR
img_path = cfg.Breastdata.ROOT_DIR + 'JPEGImages'
feature_path = cfg.Breastdata.ROOT_DIR + 'Features'

ori_anno_path = cfg.Breastdata.ROOT_DIR + 'Annotations'
set_path = cfg.Breastdata.SET_SPLIT
cache_path = cfg.CACHE_PATH

KPT_NAMES = {
    'breast': ['mass', 'nipple', 'chest', 'skin']
}


class Breast:
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        self.classes = cfg.Breastdata.CLASSES
        self.kpt_len = [len(KPT_NAMES[_]) for _ in cfg.Breastdata.CLASSES]

        self.classes_kpts = {cls: len(KPT_NAMES[cls]) for cls in self.classes}

        self.anno_path = Path(anno_path)
        self.img_path = Path(img_path)
        self.feature_path = Path(feature_path)
        self.ori_anno_path = Path(ori_anno_path)
        self.obj_resize = obj_resize
        self.sets = sets

        assert sets == 'train' or 'test', 'No match found for dataset {}'.format(sets)
        cache_name = 'breast_db_' + sets + '.pkl'
        self.cache_path = Path(cache_path)
        self.cache_file = self.cache_path / cache_name
        if self.cache_file.exists():
            with self.cache_file.open(mode='rb') as f:
                self.xml_list = pickle.load(f)
                # print("type(self.xml_list)", type(self.xml_list))
                # self.xml_list = self.xml_list.tolist()
                # print("type(self.xml_list)", type(self.xml_list))
            print('xml list loaded from {}'.format(self.cache_file))
        else:
            print('Caching xml list to {}...'.format(self.cache_file))
            self.cache_path.mkdir(exist_ok=True, parents=True)
            with np.load(set_path, allow_pickle=True) as f:
                self.xml_list = f[sets]
                # print("type(self.xml_list)", type(self.xml_list))
                self.xml_list = self.xml_list.tolist()
                # print("type(self.xml_list)", type(self.xml_list))
                # print(self.xml_list[0])
                # print(self.xml_list)

            before_filter = sum([len(k) for k in self.xml_list])

            #new breast data have already filtered
            # self.filter_list()
            after_filter = sum([len(k) for k in self.xml_list])
            with self.cache_file.open(mode='wb') as f:
                pickle.dump(self.xml_list, f)
            print('Filtered {} images to {}. Annotation saved.'.format(before_filter, after_filter))

    def filter_list(self):
        """
        Filter out 'truncated', 'occluded' and 'difficult' images following the practice of previous works.
        In addition, this dataset has uncleaned label (in person category). They are omitted as suggested by README.
        """
        for cls_id in range(len(self.classes)):
            to_del = []
            for xml_name in self.xml_list[cls_id]:
                xml_comps = xml_name.split('/')[-1].strip('.xml').split('_')
                ori_xml_name = '_'.join(xml_comps[:-1]) + '.xml'
                voc_idx = int(xml_comps[-1])
                xml_file = self.ori_anno_path / ori_xml_name
                assert xml_file.exists(), '{} does not exist.'.format(xml_file)
                tree = ET.parse(xml_file.open())
                root = tree.getroot()
                obj = root.findall('object')[voc_idx - 1]

                difficult = obj.find('difficult')
                if difficult is not None: difficult = int(difficult.text)
                occluded = obj.find('occluded')
                if occluded is not None: occluded = int(occluded.text)
                truncated = obj.find('truncated')
                if truncated is not None: truncated = int(truncated.text)
                if difficult or occluded or truncated:
                    to_del.append(xml_name)
                    continue

                # Exclude uncleaned images
                if self.classes[cls_id] == 'person' and int(xml_comps[0]) > 2008:
                    to_del.append(xml_name)
                    continue

                # Exclude overlapping images in Willow
                #if self.sets == 'train' and (self.classes[cls_id] == 'motorbike' or self.classes[cls_id] == 'car') \
                #        and int(xml_comps[0]) == 2007:
                #    to_del.append(xml_name)
                #    continue

            for x in to_del:
                self.xml_list[cls_id].remove(x)

    def get_pair(self, cls=None, shuffle=True):
        """
        Randomly get a pair of objects from VOC-Berkeley keypoints dataset
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :return: (pair of data, groundtruth permutation matrix)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_pair = []
        xml_name_pair = []

        for xml_name in random.sample(self.xml_list[cls], 2):
            anno_dict = self.__get_anno_dict(xml_name, cls)
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_pair.append(anno_dict)
            xml_name_pair.append(xml_name)

        perm_mat = np.zeros([len(_['keypoints']) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        mass_row_index = []
        mass_col_index = []
        for i, keypoint in enumerate(anno_pair[0]['keypoints']):
            for j, _keypoint in enumerate(anno_pair[1]['keypoints']):
                if keypoint['name'] == _keypoint['name']:
                    perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    if keypoint['name'] == "mass":
                        mass_row_index.append(i)
                        mass_col_index.append(j)
                    break
        row_list.sort()
        col_list.sort()
        perm_mat = perm_mat[row_list, :]
        perm_mat = perm_mat[:, col_list]
        anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]

        mass_pair_index = np.stack((mass_row_index, mass_col_index), axis=0)

        # print("mass_pair_index", mass_pair_index)

        # print("xml_name_pair",xml_name_pair)
        # print("row_list", row_list)
        # print("col list", col_list)
        # print("anno_pair[0]['keypoints']", anno_pair[0]['keypoints'])
        # print("anno_pair[1]['keypoints']", anno_pair[1]['keypoints'])


        keypoint_position_0 = {}
        for row_list_index in range(len(row_list)):
            position_x = anno_pair[0]['keypoints'][row_list_index]['x']
            position_y = anno_pair[0]['keypoints'][row_list_index]['y']
            position = [position_x, position_y]
            keypoint_position_0[row_list_index] = position


        distance_feat_0 = {}
        for dis_i in range(len(row_list)):
            distance_feat_temp = []
            for dis_j in range(len(row_list)):
                if dis_i != dis_j:
                    distance = np.sqrt((keypoint_position_0[dis_i][0] - keypoint_position_0[dis_j][0])**2 + (keypoint_position_0[dis_i][1] - keypoint_position_0[dis_j][1])**2)
                    distance_feat_temp.append(distance)
            distance_feat_0[dis_i] = distance_feat_temp

        feature_file_list = []
        feature_file_list2 = []
        for row_list_index in range(len(row_list)):

            region_name = anno_pair[0]['keypoints'][row_list_index]['name']
            # print("region_name",region_name)

            region_name_total = region_name + ".npy"
            # print("region_name_total",region_name_total)

            region_name_path = xml_name_pair[0].replace('1.xml', region_name_total)
            region_name_path = region_name_path.replace('breast/', '')
            # print("region_name_path", region_name_path)

            feature_file_path = self.feature_path / region_name_path
            # feature_file_path = "./data/Breast/Breastdata/Features/" + region_name_path
            # print("feature_file_path", feature_file_path)


            feature_file = np.load(feature_file_path)   #1*1000   [[]]



            edge_feature = distance_feat_0[row_list_index]    #1*2

            # print("edge feature", edge_feature)
            #
            # print("feature_file", feature_file)
            #
            # print("np.array([edge_feature])", np.array([edge_feature]))

            feature_file = np.concatenate((feature_file, np.array([edge_feature])), axis=1)

            # print("feature_file",feature_file)

            # feature_file = [feature_file, np.array(edge_feature)]

            feature_file_list.append(feature_file)
            # print("len feature_file", len(feature_file[0]))

        anno_pair[0]['feats'] = feature_file_list
        # print("len feature_file_list", len(feature_file_list))





        keypoint_position = {}
        for col_list_index in range(len(col_list)):
            position_x = anno_pair[1]['keypoints'][col_list_index]['x']
            position_y = anno_pair[1]['keypoints'][col_list_index]['y']
            position = [position_x, position_y]
            keypoint_position[col_list_index] = position


        distance_feat = {}
        for dis_i in range(len(col_list)):
            distance_feat_temp = []
            for dis_j in range(len(col_list)):
                if dis_i != dis_j:
                    distance = np.sqrt((keypoint_position[dis_i][0] - keypoint_position[dis_j][0])**2 + (keypoint_position[dis_i][1] - keypoint_position[dis_j][1])**2)
                    distance_feat_temp.append(distance)
            distance_feat[dis_i] = distance_feat_temp



        for col_list_index in range(len(col_list)):

            region_name_2 = anno_pair[1]['keypoints'][col_list_index]['name']
            # print("region_name_2",region_name_2)

            region_name_total_2 = region_name_2 + ".npy"
            # print("region_name_total_2",region_name_total_2)

            region_name_path_2 = xml_name_pair[1].replace('1.xml', region_name_total_2)
            region_name_path_2 = region_name_path_2.replace('breast/', '')

            # print("region_name_path_2", region_name_path_2)

            feature_file_path_2 = self.feature_path / region_name_path_2
            # print("feature_file_path_2", feature_file_path_2)

            feature_file_2 = np.load(feature_file_path_2)

            edge_feature = distance_feat[col_list_index]

            # print("edge feature", edge_feature)

            feature_file_2 = np.concatenate((feature_file_2, np.array([edge_feature])), axis=1)
            # feature_file_2 = [feature_file_2, np.array(edge_feature)]

            # feature_file_2
            feature_file_list2.append(feature_file_2)
            # print("feature_file_2", len(feature_file_2[0]))

        anno_pair[1]['feats'] = feature_file_list2
        # print("len feature_file_list2", len(feature_file_list2))


        # print("anno_pair[0] ", anno_pair[0]['keypoints'])
        # print("anno_pair[1] ", anno_pair[1]['keypoints'])


        return anno_pair, perm_mat, mass_pair_index

    def __get_anno_dict(self, xml_name, cls):
        """
        Get an annotation dict from xml file
        """
        xml_file = self.anno_path / xml_name
        assert xml_file.exists(), '{} does not exist.'.format(xml_file)


        # print("xml_file", xml_file)

        tree = ET.parse(xml_file.open())
        root = tree.getroot()

        img_name = root.find('./image').text + '.jpg'
        img_file = self.img_path / img_name


        # print("img_name", img_name)
        # print("img_file", img_file)


        bounds = root.find('./visible_bounds').attrib
        h = float(bounds['height'])
        w = float(bounds['width'])
        xmin = float(bounds['xmin'])
        ymin = float(bounds['ymin'])
        with Image.open(str(img_file)) as img:
            ori_sizes = img.size
            obj = img.resize(self.obj_resize, resample=Image.BICUBIC, box=(xmin, ymin, xmin + w, ymin + h))

        keypoint_list = []
        for keypoint in root.findall('./keypoints/keypoint'):
            attr = keypoint.attrib
            attr['x'] = (float(attr['x']) - xmin) * self.obj_resize[0] / w
            attr['y'] = (float(attr['y']) - ymin) * self.obj_resize[1] / h
            keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict['image'] = obj
        anno_dict['keypoints'] = keypoint_list
        anno_dict['bounds'] = xmin, ymin, w, h
        anno_dict['ori_sizes'] = ori_sizes
        anno_dict['cls'] = self.classes[cls]

        # print("anno_dict['image']", anno_dict['image'])
        # print("anno_dict['keypoints']", anno_dict['keypoints'])

        return anno_dict

if __name__ == '__main__':
    dataset = Breast('train', (256, 256))
    a = dataset.get_pair()
    pass