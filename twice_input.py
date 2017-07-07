import numpy as np
import os
from PIL import Image
from tflearn.data_utils import shuffle

dataList = list()

directories = os.listdir('data')
for directory in directories:
    dataList.append([np.array(Image.open('data/' + directory + '/' + file)).flatten() for file in os.listdir('data/' + directory)])

print(dataList)




