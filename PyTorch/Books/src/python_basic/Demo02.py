import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Parent:

    def __init__(self):
        pass

    def print_info(self):
        print("This is Parent.")


class Child(Parent):

    def __init__(self):
        pass

    def print_info(self):
        print("This is Child.")


child = Child()
child.print_info()