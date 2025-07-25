import torch
import torch.nn as nn

a = torch.rand(2,3)
print(a, type(a), a.size(), a.shape, '\n', nn.Sigmoid()(a))

import os, sys
print(os.path.split('~Flispark/code/backend/file.txt'))
print(os.path.splitext('~Flispark/code/backend/file.txt')); sys.exit(0)

# import kagglehub
# path = kagglehub.dataset_download("pes1ug22am047/damaged-and-undamaged-artworks")
# print("Path to dataset files:", path)


import matplotlib.pyplot as plt
categories = ['Apples', 'Bananas', 'Oranges', 'Grapes']
sales = [150, 200, 120, 180]

plt.bar(categories, sales)
plt.title('Fruit Sales Data')
plt.xlabel('Fruit Type')
plt.ylabel('Number of Sales')
plt.show()


