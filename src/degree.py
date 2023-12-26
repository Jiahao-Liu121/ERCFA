import os
import numpy as np

kg = '../data/book/kg_final'
if os.path.exists(kg + '.npy'):
    rating_np = np.load(kg + '.npy')
else:
    rating_np = np.loadtxt(kg + '.txt', dtype=np.int64)
    np.save(kg + '.npy', rating_np)

a = max(set(rating_np[:, 0]))
b = max(set(rating_np[:, 2]))
if a >= b:
    c = a
else:
    c = b
num = []
for i in range(c + 1):
    num.append(0)
f = open('../data/book/kg_final.txt', 'r')
for line in f.readlines():
    line = line.split()
    num[int(line[0])] += 1
    num[int(line[2])] += 1

for i in range(len(num)):
    if num[i] > 1000:
        num[i] = 999
