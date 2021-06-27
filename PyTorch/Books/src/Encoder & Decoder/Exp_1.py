import numpy as np


list = [1, 2, 3]
print("list:", list)
print("type(list):", type(list))
print("*list:", *list)
# print("type(*list):", type(*list))  # Exception

print("-" * 50)

newTuple = (1, 2, 3)
print("newTuple:", newTuple)
print("type(newTuple):", type(newTuple))
print("*newTuple:", *newTuple)
# print("type(*newTuple):", type(*newTuple))  # Exception

print("-" * 50)

np.random.seed(42)

for i in np.random.randn(1, 2, 3):
    print("i:")
    print(i)

print("-" * 50)
newRandn = np.random.randn(*newTuple)
print("newRandn:")
print(newRandn)

print("-" * 50)
a = np.random.randn(1)
b = np.random.randn(2)
c = np.random.randn(3)
print(a)
print(b)
print(c)

"""
Res:
    list: [1, 2, 3]
    type(list): <class 'list'>
    *list: 1 2 3
    --------------------------------------------------
    newTuple: (1, 2, 3)
    type(newTuple): <class 'tuple'>
    *newTuple: 1 2 3
    --------------------------------------------------
    i:
    [[ 0.49671415 -0.1382643   0.64768854]
     [ 1.52302986 -0.23415337 -0.23413696]]
    --------------------------------------------------
    newRandn:
    [[[ 1.57921282  0.76743473 -0.46947439]
      [ 0.54256004 -0.46341769 -0.46572975]]]
    --------------------------------------------------
    [0.24196227]
    [-1.91328024 -1.72491783]
    [-0.56228753 -1.01283112  0.31424733]
"""