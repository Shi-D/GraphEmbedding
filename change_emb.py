#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022-02-19 13:16 
# @Author : shi-d
# @File : hfiuhusd.py
# @Software: PyCharm
# @GitHub : Shi-D
# @Homepage : https://shi-d.github.io/

import numpy as np


file_handler = open('../data/BA/ba.emb', mode='r')
emb_content = file_handler.readlines()
emb_lists = []
for emb in emb_content:
    emb = emb.strip('\n')
    emb = emb.split(' ')
    emb[0] = int(emb[0])
    emb[1] = emb[1][1:]

    for i, e in enumerate(emb):
        if i != 0:
            emb[i] = float(emb[i][0:-1])

    emb_lists.append(emb)

print(emb_lists[:10])
emb_lists = [value for index, value in sorted(enumerate(emb_lists), key=lambda emb_lists:emb_lists[1])]
print(emb_lists[:10])

with open('../data/BA/ba.emb_8', 'w') as f:
    for emb in emb_lists:
        s = ''
        for i in emb:
            s = s + str(i)+' '
        s = s[0:-1]
        f.writelines(s+'\n')