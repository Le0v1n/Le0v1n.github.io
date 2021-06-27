import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

a = np.random.randn(30)
b = np.random.randn(30)
c = np.random.randn(30)
d = np.random.randn(30)

plt.subplot(2, 2, 1)
A = plt.plot(a, "r--o")[0]

plt.subplot(2, 2, 2)
B = plt.plot(b, "b-*")[0]

plt.subplot(2, 2, 3)
C = plt.plot(c)[0]

plt.subplot(2, 2, 4)
D = plt.plot(d)[0]

plt.legend([A, B, C, D], ["A", "B", "C", "D"])
plt.show()