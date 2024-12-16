import numpy as np


file = 'pavpu_mobilenet.txt'

fp =  open(file,'r')

vals = [float(line) for line in fp.readlines()]

print(np.std(vals))