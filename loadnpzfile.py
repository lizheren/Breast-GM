import numpy as np

csv_file = "./../PA-GM/data/voc2011_pairs.npz"
data = np.load(csv_file, allow_pickle=True)

test = data['test']
train = data['train']
print(type(test))
print(test[0][0])
print(test[0])
print(type(test[0]))