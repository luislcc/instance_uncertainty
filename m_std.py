import json
import os
import numpy as np
import re
from scipy.stats import ttest_ind

files = os.listdir()
div = 0
vals = { i: [] for i in range(20)}
for file in files:
    if re.match('seed[0-9]+.*json$',file):
        print(file)
        f = json.load(open(file))
        for k in vals:
            vals[k] += [f[str(k)]]

means = {i: np.mean(vals[i]) for i in range(20)}
stds = {i: np.std(vals[i]) for i in range(20)}

print(means)
print(stds)
