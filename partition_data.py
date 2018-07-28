import numpy as np
import os
import matplotlib.pyplot as plt
import random
import scipy.misc
import imageio



full_list = os.listdir('full') # get each category downloaded

# defining the base paths
train_path = 'train/'
val_path = 'validation/'
test_path = 'test/'

# creating a folder for every category for training, validation, and testing sets
for item in full_list:
    category = item.replace('.npy', '')
    category = category.replace(' ', '_')
    path1 = train_path + category
    path2 = val_path + category
    path3 = test_path + category
    
    if not os.path.exists(path1):
        os.makedirs(path1)
        
    if not os.path.exists(path2):
        os.makedirs(path2)

    if not os.path.exists(path3):
        os.makedirs(path3)


# randomly converting 52500 of each category to images
# 45000 training
# 2500 validaiton
# 2500 testing


training_size = 50000
val_size = 5000
# test_size = 2500


for idx, item in enumerate(full_list):
    file = np.load('full/{}'.format(item))
    rand_idx = random.sample(range(file.shape[0]), training_size + val_size
                                                    # + test_size
        )
    file = file.reshape(file.shape[0], 28, 28)
    
    category = item.replace('.npy', '')
    category = category.replace(' ', '_')
    
    print('\nWorking on category {}/{}'.format(idx+1, len(full_list)))

    train = file[rand_idx[:training_size]]
    print('Currently saving train/{}'.format(category))
    for num, img in enumerate(train):
        path = '{}{}/{}{}.png'.format(train_path,category,category,str(num))
        if not os.path.exists(path):
            imageio.imwrite(path, img)

    print('Currently saving validation/{}'.format(category))
    val = file[rand_idx[training_size:training_size + val_size]]
    for num, img in enumerate(val):
        path = '{}{}/{}{}.png'.format(val_path,category,category,str(num))
        if not os.path.exists(path):
            imageio.imwrite(path, img)

    # print('Currently saving test/{}'.format(category))
    # test = file[rand_idx[training_size + val_size:]]
    # for num, img in enumerate(test):
    #     path = '{}{}/{}{}.png'.format(test_path,category,category,str(num))
    #     if not os.path.exists(path):
    #         imageio.imwrite(path, img)

print('\nDone!')
