from pathlib import Path
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import random
import pickle
import torch
import sys
import os



anno_path = './annotations/breast/'

filenamelist = os.listdir(anno_path)


for i in range(len(filenamelist)):
    xml_file = anno_path + filenamelist[i]
    xml_file = Path(xml_file)
    assert xml_file.exists(), '{} does not exist.'.format(xml_file)

    print("xml_file", xml_file)

    tree = ET.parse(xml_file.open())
    root = tree.getroot()