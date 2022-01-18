import numpy as np
import os
from skimage import io,data
import matplotlib.pyplot as plt


path = "./../TMI_mass_matching/multi_view_numpy/"
filenamelist = os.listdir(path)


for i in range(len(filenamelist)):
    load_npy_file_path = "./../TMI_mass_matching/multi_view_numpy/" + filenamelist[i]
    print(load_npy_file_path)
    file = np.load(load_npy_file_path)


    image_save_path_root = "./../TMI_mass_matching/breast_image/"
    image_save_path = image_save_path_root + str(filenamelist[i]).replace('.npy', '') + '.jpg'

    print(image_save_path)


    # img = data.coffee()
    io.imshow(file)  # 显示图片
    io.imsave(image_save_path, file) # 保存图片
    plt.show()