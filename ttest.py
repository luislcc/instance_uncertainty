import json
import os
import numpy as np
import re
from scipy.stats import ttest_ind

classes = [6,11,12,17,18]

files = os.listdir('unc_iou_acdc2cs')
div = 0
vals = { i: [] for i in classes}
for file in files:
    #if re.match('seed[0-9]+.*json$',file):
    f = json.load(open(os.path.join('unc_iou_acdc2cs',file)))
    for k in vals:
        vals[k] += [f[str(k)]]

means = {i: np.mean(vals[i]) for i in classes}
stds = {i: np.std(vals[i]) for i in classes}

#print(means)

files2 = os.listdir()

vals2 = {i: [] for i in classes}

for file2 in files2:
    if re.match('seed[0-9]+.*json$',file2):
        f = json.load(open(file2))
        for k in vals2:
            vals2[k] += [f[str(k)]]

p_values = {i: ttest_ind(vals[i],vals2[i])[1] for i in classes}

print(p_values)