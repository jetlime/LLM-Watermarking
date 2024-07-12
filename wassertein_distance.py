import matplotlib.pylab as plt
import numpy as np
import os
from scipy.stats import wasserstein_distance
import itertools

def read_distribution_from_file(file_name):
    list_ = []
    with open(file_name, "r") as f:
        for line in f:
            list_.append(int(line.strip()))
    return list_

files = os.listdir("./results/")
file_names = [file for file in files if file.startswith('result_')]

distributions_human = []
distributions_llm = []

for file_name in file_names:
    if file_name.split('_')[1] == "human":
        distributions_human.append(read_distribution_from_file(f"./results/{file_name}"))
    else:
        distributions_llm.append(read_distribution_from_file(f"./results/{file_name}"))

print(distributions_human)
print(distributions_llm)

# Do metric computation and plotting below
# plt.figure(1)
# plt.plot(xs[:, 0], xs[:, 1], '+b', label='Human Samples')
# plt.plot(xt[:, 0], xt[:, 1], 'xr', label='LLM Samples')
# plt.legend(loc=0)
# plt.title('LLM and Human Text Generated distributions')
# plt.savefig("./results/overall_wassestein_distance.png")