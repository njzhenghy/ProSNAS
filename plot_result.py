import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import os

plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
font ={'family':'sans-serif',
       'weight': 'normal',
       'size': 42}


alldata = [[] for i in range(10)]


data_list = ['codrna']
# data_list = ['mnist']
# data_list = ['madelon']
fig = plt.figure(figsize=(13, 12))
alldata = [[] for i in range(10)]
time = [[] for i in range(10)]
path = './CIFAR00/'
dirs = os.listdir(path)
for j, dir  in enumerate(dirs):
    dir_path = os.path.join(path, dir)
    files = os.listdir(dir_path)
    for i, fs in enumerate(files):
        dirs = os.path.join(dir_path, fs)
        for k, f in enumerate(os.listdir(os.path.join(dir_path, fs))):
            if f == 'log.txt':
                with open(os.path.join(dirs, f), 'r') as file:
                    val_acc = []
                    while True:
                        line = file.readline()
                        if not line:
                            break
                        words = line.split(' ')
                        if words[2] == 'valid_acc':
                            val_acc.append(float(words[3].split('\n')[0]))
                        # 检查是否到达文件末尾
                        
        alldata[j].append(val_acc)
        time[j].append(list(range(len(val_acc))))
    alldata[j] = np.array(alldata[j])
    time[j] = np.array(time[j]).mean(axis=0)

    alg_name = dir.split('_')[0]
    color = palette(j)
    avg = np.mean(alldata[j], axis=0)
    std = np.std(alldata[j], axis=0)
    r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
    r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
    plt.plot(time[j], avg, color=color, label=alg_name, linewidth=4.0)
    plt.fill_between(time[j], r1, r2, color=color, alpha=0.2)
    print(' alg %s avg %f std %f'%( alg_name, avg[-1], std[-1]))

plt.tick_params(labelsize=32)
plt.legend(loc='lower right', prop=font)
plt.xlabel('Iterations', font=font)
# plt.xlim((-10,300))
# plt.xlim((-50, 3500))
# plt.ylim((0.65, 1.05))
plt.ylabel('Validation Accuracy', font=font)
# plt.savefig(save_folder+'/time_acc.jpeg')
plt.savefig('./img/search'  + '.pdf',dpi=600)
# plt.show()