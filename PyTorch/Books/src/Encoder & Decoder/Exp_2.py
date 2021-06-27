import numpy as np

a = (1, 2, 3)
np.random.seed(42)
A = np.random.randn(*a)
np.random.seed(42)
B = np.random.randn(1, 2, 3)

print("A:")
print(A)
print("B:")
print(B)

"""
Res:
    A:
    [[[ 0.49671415 -0.1382643   0.64768854]
      [ 1.52302986 -0.23415337 -0.23413696]]]
    B:
    [[[ 0.49671415 -0.1382643   0.64768854]
      [ 1.52302986 -0.23415337 -0.23413696]]]
"""