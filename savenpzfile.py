import numpy as np
import os

path = './annotations/breast/'
filenamelist = os.listdir(path)

# print(len(filenamelist))
# print(filenamelist[0])
#
# print(filenamelist)
train_list_0 = []
test_list_0 = []

for i in range(int(0.8* len(filenamelist))):
    filename = 'breast/' + str(filenamelist[i])

    train_list_0.append(str(filename))

# train_array = np.array([])
# np.insert(train_array, list(train_list_0))
train_array = np.array([list(train_list_0)])

for i in range(int(0.8* len(filenamelist)), len(filenamelist)):
    filename = 'breast/' + str(filenamelist[i])

    test_list_0.append(str(filename))


# test_array = np.array([])
# np.insert(test_array, list(test_list_0))
test_array = np.array([list(test_list_0)])

np.savez('./breast_pairs.npz', train=train_array, test=test_array)




csv_file = "./Breastdata_pairs.npz"
data = np.load(csv_file, allow_pickle=True)

test = data['test']
train = data['train']
print(type(test))
print(test[0][0])
print(test[0])
print(type(test[0]))
