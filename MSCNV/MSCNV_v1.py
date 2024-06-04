import sys

import numpy as np
import pysam
import os
import math
import random
import csv
import time
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
# 检测数据集中异常值的模型
from pyod.models.ocsvm import OCSVM

def get_chrlist(filename):
    # 从bam文件中读取染色体序列，查看bam文件中有几条染色体，在仿真数据中仅有21号染色体
    # 输入：bam 输出：染色体的列表
    samfile = pysam.AlignmentFile(filename, "rb")
    List = samfile.references
    chrList = np.full(len(List), 0)
    for i in range(len(List)):
        chr = str(List[i]).strip('chr')
        if chr.isdigit():
            chrList[i] = int(chr)
    return chrList

def get_RC(filename, chrList, ReadCount, mapq):
    # 从bam文件中提取每个位点的read count值，建议打开sam文件，查看sam文件的格式
    # 输入：bam文件，染色体的列表和初始化的rc数组 输出：rc数组
    samfile = pysam.AlignmentFile(filename, "rb")
    for line in samfile:
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                num = np.argwhere(chrList == int(chr))[0][0]
                # num = 0
                posList = line.positions
                ReadCount[num][posList] += 1
                mapq[num][posList] += line.mapq
    return ReadCount, mapq

def read_ref_file(filename, ref, num):
    # 读取reference文件
    # 输入：fasta文件， 初始化的ref数组 和 当前染色体号  输出： ref数组
    if os.path.exists(filename):
        print("Read reference file: " + str(filename))
        with open(filename, 'r') as f:
            line = f.readline()
            for line in f:
                linestr = line.strip()
                ref[num] += linestr
    else:
        print("Warning: can not open " + str(filename) + '\n')
    return ref

def get_mapq(filename, chrList, mapq,chr):
    # get mapq for each position
    # input：bam file, the list of Chromosomes and rc array. output：rc array
    samfile = pysam.AlignmentFile(filename, "rb")
    for line in samfile:
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                # num = np.argwhere(chrList == int(chr))[0][0]
                num = 0
                posList = line.positions
                mapq[num][posList] += line.mapq
    return mapq

def ReadDepth(mapq,ReadCount, binNum, ref,binSize):
    # get read depth
    '''
       1. compute the mean of rc in each bin;
       2. count the number of 'N' in ref. If there is a 'N' in a bin，the rd is not counted;
       3. GC bias
    '''

    RD = np.full(binNum, 0.0)
    GC = np.full(binNum, 0)
    MQ = np.full(binNum, 0.0)
    pos = np.arange(1, binNum+1)
    for i in range(binNum):
        RD[i] = np.mean(ReadCount[i*binSize:(i+1)*binSize])
        MQ[i] = np.mean(mapq[i*binSize:(i+1)*binSize])
        cur_ref = ref[i*binSize:(i+1)*binSize]
        N_count = cur_ref.count('N') + cur_ref.count('n')
        if N_count == 0:
            gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
        else:
            RD[i] = -10000
            gc_count = 0
        GC[i] = int(round(gc_count / binSize, 3) * binSize)

    index = RD > 0
    RD = RD[index]
    GC = GC[index]
    MQ = MQ[index]
    pos = pos[index]
    RD = gc_correct(RD, GC)
    return pos, RD, MQ ,GC

# def ReadDepth(ReadCount, binNum, ref):
#     # 从rc数组中获取每个bin的read depth值
#     '''
#        1.计算每个bin中read count的平均值
#        2.统计ref对应每个bin中N的个数，只要一个bin中有一个N，就不统计该bin的rd值
#        3.对rd进行GC bias纠正
#     '''
#
#     RD = np.full(binNum, 0.0)
#     GC = np.full(binNum, 0)
#     pos = np.arange(1, binNum+1)
#     for i in range(binNum):
#         RD[i] = np.mean(ReadCount[i*binSize:(i+1)*binSize])
#         cur_ref = ref[i*binSize:(i+1)*binSize]
#         N_count = cur_ref.count('N') + cur_ref.count('n')
#         if N_count == 0:
#             gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
#         else:
#             RD[i] = -10000
#             gc_count = 0
#         GC[i] = int(round(gc_count / binSize, 3) * 1000)
#
#     index = RD > 0
#     RD = RD[index]
#     GC = GC[index]
#     pos = pos[index]
#     RD = gc_correct(RD, GC)
#
#     return pos, RD

def gc_correct(RD, GC):
    # correcting gc bias
    bincount = np.bincount(GC)
    global_rd_ave = np.mean(RD)
    for i in range(len(RD)):
        if bincount[GC[i]] < 2:
            continue
        mean = np.mean(RD[GC == GC[i]])
        RD[i] = global_rd_ave * RD[i] / mean
    return RD


def scaling_RD(RD, mode):   # 缩放校正，因为segmentation的时候会将信号改变，所以这一步将信号加强，强的信号放大，弱信号缩小。
    posiRD = RD[RD > mode]
    negeRD = RD[RD < mode]
    if len(posiRD) < 50:
        mean_max_RD = np.mean(posiRD)
    else:
        sort = np.argsort(posiRD)
        maxRD = posiRD[sort[-50:]]  # posiRD中最大的50个
        mean_max_RD = np.mean(maxRD)

    if len(negeRD) < 50:
        mean_min_RD = np.mean(negeRD)
    else:
        sort = np.argsort(negeRD)
        minRD = negeRD[sort[:50]]   # negeRD中最小的50个
        mean_min_RD = np.mean(minRD)
    scaling = mean_max_RD / (mode + mode - mean_min_RD)  # 规定一个值  1.1269921174554123
    # print(scaling)
    for i in range(len(RD)):
        if RD[i] < mode:
            RD[i] /= scaling        # 让loss信号变的更低
    # print('RD:',RD)
    return RD


def modeRD(RD):             # 返回RD中最密集的一个值
    newRD = np.full(len(RD), 0)
    for i in range(len(RD)):    # RD就是每个bin中，所有RC之和除以bin的长度
        # print(RD[i])
        newRD[i] = int(round(RD[i], 3) * 1000)  # RD值保留三位小数并*1000变成整数，bincount()需要整数
    count = np.bincount(newRD)  # count存放 count下标在newRD中出现的次数
    # count的长度比newRD中的最大值大1
    countList = np.full(len(count) - 49, 0)     # 因为下面每50值要算平均值，此时newRD是百位数，所以50不算太大，对应原RD的小数位
    for i in range(len(countList)):
        # 一次向后移动一格 计算下一块的50个数值之和
        countList[i] = np.mean(count[i:i + 50])  # countList存放50个newRD值出现次数之和的平均值
    modemin = np.argmax(countList)
    modemax = modemin + 50
    mode = (modemax + modemin) / 2  # 50个最大的ReadDepth的首尾下标，然后得到中间下标，即出现
        # 次数最多的ReadDepth的中间值
    mode = mode / 1000              # 上边乘1000，这里要除以
    return mode


def plot(pos, data):
    plt.scatter(pos, data, s=3, c="black")
    plt.xlabel("pos")
    plt.ylabel("rd")
    plt.show()

def plotRDMQ(RD, MQ):
    #plot RD and MQ
    x_value = range(1, len(RD) + 1)
    plt.scatter(x_value, MQ)
    plt.scatter(x_value, RD)
    plt.show()

def data_norm(scores):
    '''对生成的异常分数进行归一化处理'''
    ma = scores.max()
    mi = scores.min()
    for i in range(len(scores)):
        scores[i] = (scores[i] - mi) / (ma - mi)
    return(scores)

def var(S,D0,D1,threshold):#计算每个阈值下的类间方差
    noridx = S <= threshold #正常点索引
    unnoridx = S > threshold #异常点索引
    if len(D0) == 0:
        avg_s0 = 0
    else:
        avg_s0 = sum(S[noridx])/len(D0)
    if len(D1) == 0:
        avg_s1 = 0
    else:
        avg_s1 = sum(S[unnoridx])/len(D1)
    p0 = len(noridx)/len(S)
    p1 = len(S)-p0
    avg_global = p0*avg_s0 + p1*avg_s1
    P0 = math.pow(avg_s0-avg_global,2)
    P1 = math.pow(avg_s1-avg_global,2)
    var = p0*P0 + p1*P1
    return var

def split_data(S,threshold):
    D0 = []
    D1 = []
    for i in range(len(S)):
      if S[i] < threshold:
          D0.append(S[i])# 正常样本集
      else:
          D1.append(S[i])# 异常样本集
          return D0, D1

def subsample(D,len):
    D = random.sample(D, len)
    return D

def choice_threshold(S,low,high):
    for i in range(len(S)):
        S[i] = round(S[i]*100)/100 # 将S(异常分数)保留两位小数
    final_threshold = np.median(S) # 最终阈值 初始化为异常分数的 中位数
    max_var = 0.0
    low = round(low,2)
    high = round(high,2)
    for i in np.arange(low,high,0.01):
        cur_threshold = round(i,2)
        D0 , D1 = split_data(S,cur_threshold)
        if len(D0) > len(D1):
            D0 = subsample(D0,len(D1))
        else:
            D1 = subsample(D1,len(D0))
        cur_var = var(S,D0,D1,cur_threshold)
        if cur_var > max_var:
            final_threshold = cur_threshold
            max_var = cur_var
    return final_threshold


def merge_bin(labels, maxbin, pos,binSize,RD,MQ,scores):                              #还需要引入pem测略来进一步精确pos区域,目前的区域假阳性太多
    # merge bin
    start = ends = step = 0
    cnv_range = []
    # rd_index = []
    for i in range (len(labels)):
        if labels[i]== 1 and start == 0:#如果为异常点且开始为0,则设置start的位置
            start = pos[i] * binSize
            rd_index_start = i
            continue
        if labels[i] == 1 and start != 0:#如果为异常点且开始不为0,则设置end的位置
            ends = pos[i] * binSize
            rd_index_end = i
            # rd_index.append([rd_index_start,rd_index_end])
            step = 0
            continue
        if labels[i] != 1 and start != 0:#如果为正常点且开始不为0
            if (i == len(labels)-1):
                cnv_range.append([start * binSize, ends * binSize,np.mean(RD[rd_index_start:rd_index_end+1]),np.mean(MQ[rd_index_start:rd_index_end+1]),np.mean(scores[rd_index_start:rd_index_end+1])])
            step = step + 1
            # maxbin = 3
            if step == maxbin:
                if (ends - start >= maxbin - 2):
                    cnv_range.append([start* binSize,ends* binSize,np.mean(RD[rd_index_start:rd_index_end+1]),np.mean(MQ[rd_index_start:rd_index_end+1]),np.mean(scores[rd_index_start:rd_index_end+1])])
                start = 0
                ends = 0
                step = 0
    return cnv_range

def get_exact_cnv_position(reference,binSize,bam,cnv_range):
    exact_cnv_position = []
    disRange = np.empty(len(cnv_range), dtype=object)
    bf = pysam.AlignmentFile(bam, 'rb')
    maxbin = 3
    # print(len(cnv_range))
    for i in range(len(cnv_range)):
        disRange[i] = []
        if cnv_range[i][0] - maxbin * binSize > cnv_range[i][1] + maxbin * binSize:  # 如果左边比右边大,则跳出本次循环
            continue
        for r in bf.fetch(str(reference.split('/')[-1].split('.fa')[0]), cnv_range[i][0] - maxbin * binSize,
                          cnv_range[i][1] + maxbin * binSize):
            r_size = cnv_range[i][1]-cnv_range[i][0]
            # r_sizeMin = r_size - 100
            # r_sizeMax = r_size + 100
            #
            # if abs(r.isize) > r_sizeMax or abs(r.isize) < r_sizeMin: #如果变异区间和插入长度相等,则删除这个变异区间      这个变异区间是否是比对长度???
            disRange[i].append([r.reference_name, r.pos, r.cigarstring, r.isize, r_size])
        #         print(disRange)
        # print('---------')
    # print('length:',len(disRange))
    sizeresult = 'disrangesizeresult'
    with open(sizeresult, 'w') as fpem:
        for i in range(len(disRange)):
            start = 0
            end = 0
            fpem.write("\nthis is " + str(i) + " dis range:\n")
            # 将保存的信息写进这个文件中
            fpem.writelines(str(disRange[i]))
            # 基于pem策略检测串联重复和loss
            for j in range(len(disRange[i])):
                if disRange[i][j][3] > 0:
                    if not start:
                        start = disRange[i][j][1]
                if disRange[i][j][3] < 0:
                    end = disRange[i][j][1]
            if (start == 0 and end == 0):
                    continue
            if (start == 0):
                start = cnv_range[i][0]

            if (end == 0):
                end = cnv_range[i][1]
            exact_cnv_position.append([int(start),int(end)])
    return exact_cnv_position
#
# def get_exact_position(reference,binSize, bam, cnv_range):
#     # get exact position
#     bp_exact_position = []
#     not100Mperrange = np.empty(len(cnv_range), dtype=object)
#     bf = pysam.AlignmentFile(bam, 'rb')
#     # maxbin可以调节
#     maxbin = 2
#     for i in range(len(cnv_range)):
#         not100Mperrange[i] = []
#         if cnv_range[i][0] - maxbin * binSize > cnv_range[i][1] + maxbin * binSize:  # 如果左边比右边大,则跳出本次循环
#             continue
#         for r in bf.fetch(str(reference.split('/')[-1].split('.fa')[0]), cnv_range[i][0] - maxbin * binSize,
#                           cnv_range[i][1] + maxbin * binSize):#fetch()读取特定比对区域内的数据
#             # if (r.cigarstring != str1 and r.cigarstring != None):
#             if r.cigarstring:
#                 not100Mperrange[i].append([r.reference_name, r.pos, r.cigarstring, r.isize])
#     # print(not100Mperrange)
#     # 如果直接用提取出的split reads的bam文件,就不用进行上一步操作了,因为里面都是非100M的字段
#     cigarresult = "bigrangeciagrresult"
#     with open(cigarresult, 'w') as f1:
#         for i in range(len(not100Mperrange)):
#             start = 0
#             end = 0
#             # s = 0
#             # e = 2 * binSize
#
#             f1.write("\nthis is " + str(i) + " big range:\n")
#             # 将保存的信息写进这个文件中
#             f1.writelines(str(not100Mperrange[i]))
#             # 循环遍历如[['chr21', 17000166, '41S59M', 468], ['chr21', 17000166, '57H43M', 497], ['chr21', 17000166, '60H40M', 3555],
#             # ['chr21', 17000166, '45S55M', 3520], ['chr21', 17000166, '49S51M', 3562], ['chr21', 17000166, '42S58M', 3643], ['chr21', 17004114, '54M46S', -3577],
#             # ['chr21', 17004157, '60M40S', -458], ['chr21', 17004215, '5S53M42S', -3554], ['chr21', 17004222, '46M54H', -529],
#             # ['chr21', 17004226, '42M58H', -3594], ['chr21', 17004267, '53H47M', -3730], ['chr21', 17004318, '50M50H', -591], ['chr21', 17004325, '43M57H', -503]]
#             # flag = 0
#             # 基于sr策略检测串联重复和loss
#             for j in range(len(not100Mperrange[i])):
#                 # flag = 0  # flag为1的是穿插重复变异
#                 # not100Mperrange[i][j][2]代表'11S89M'这个位置
#                 # pos为M索引位置,如4
#                 # pos = not100Mperrange[i][j][2].index('M')
#
#                 p = not100Mperrange[i][j][2].split('M')
#                 if p[0].isdigit():
#                     if not end:
#                         end = not100Mperrange[i][j][1] + (int)(p[0]) - 1
#                 else:
#                     if not start:
#                         start = not100Mperrange[i][j][1]
#
#
#                 #if pos == 4 or pos == 5:
#                     # 如果start为0
#                     #if not start:
#                         # s = not100Mperrange[i][j][1]
#                         #start = not100Mperrange[i][j][1]
#                 #else:
#                     #if pos == 1 or pos == 2:
#                         # if not end:
#                         # e = not100Mperrange[i][j][1]
#                         #end = not100Mperrange[i][j][1] + (int)(not100Mperrange[i][j][2][0: pos]) - 1
#
#             # if 0 < e - s < binSize:
#             #     start = cnv_range[i][0]
#             #     end = cnv_range[i][1]
#
#             if (start == 0 and end == 0):
#                 # continue
#                 start = cnv_range[i][0]
#                 end = cnv_range[i][1]
#
#             if (start == 0):
#                 start = cnv_range[i][0]
#
#             if (end == 0):
#                 end = cnv_range[i][1]
#
#             if (start > end):
#                 temp = start
#                 start = end
#                 end = temp
#
#             if (end - start > 2 * binSize):
#                 bp_exact_position.append([int(start),int(end)])
#
#     return bp_exact_position

def get_exact_position(reference,binSize, bam, cnv_range):
    # get exact position
    bp_exact_position = []
    not100Mperrange = np.empty(len(cnv_range), dtype=object)
    bf = pysam.AlignmentFile(bam, 'rb')
    # maxbin可以调节
    maxbin = 14
    for i in range(len(cnv_range)):
        not100Mperrange[i] = []
        if cnv_range[i][0] - maxbin * binSize > cnv_range[i][1] + maxbin * binSize:  # 如果左边比右边大,则跳出本次循环
            continue

        for r in bf.fetch(str(reference.split('.fa')[0]), cnv_range[i][0] - maxbin * binSize,
                          cnv_range[i][1] + maxbin * binSize):#fetch()读取特定比对区域内的数据
            # if (r.cigarstring != str1 and r.cigarstring != None):
            if r.cigarstring:
                not100Mperrange[i].append([r.reference_name, r.pos, r.cigarstring, r.isize])
    # print(not100Mperrange)
    # 如果直接用提取出的split reads的bam文件,就不用进行上一步操作了,因为里面都是非100M的字段
    cigarresult = "bigrangeciagrresult"
    with open(cigarresult, 'w') as f1:
        for i in range(len(not100Mperrange)):
            start = 0
            end = 0

            f1.write("\nthis is " + str(i) + " big range:\n")
            # 将保存的信息写进这个文件中
            f1.writelines(str(not100Mperrange[i]))
            # 循环遍历如[['chr21', 17000166, '41S59M', 468], ['chr21', 17000166, '57H43M', 497], ['chr21', 17000166, '60H40M', 3555],
            # ['chr21', 17000166, '45S55M', 3520], ['chr21', 17000166, '49S51M', 3562], ['chr21', 17000166, '42S58M', 3643], ['chr21', 17004114, '54M46S', -3577],
            # ['chr21', 17004157, '60M40S', -458], ['chr21', 17004215, '5S53M42S', -3554], ['chr21', 17004222, '46M54H', -529],
            # ['chr21', 17004226, '42M58H', -3594], ['chr21', 17004267, '53H47M', -3730], ['chr21', 17004318, '50M50H', -591], ['chr21', 17004325, '43M57H', -503]]
            flag = 0
            # 基于sr策略检测串联重复和loss
            for j in range(len(not100Mperrange[i])):
                # flag = 0  # flag为1的是穿插重复变异
                # not100Mperrange[i][j][2]代表'11S89M'这个位置
                # pos为M索引位置,如4
                pos = not100Mperrange[i][j][2].index('M')
                # [['chr21', 10000165, '38S62M', 556], ['chr21', 10000165, '32S68M', 414],
                #  ['chr21', 10000165, '62H38M', 393], ['chr21', 10000165, '60H40M', 536],
                #  ['chr21', 10000165, '35S65M', 441], ['chr21', 10000165, '38S62M', 431],
                #  ['chr21', 10000165, '53H47M', 501], ['chr21', 10001104, '63M37S', -548],
                #  ['chr21', 10001106, '61M39S', -407], ['chr21', 10001113, '54M46S', -456],
                #  ['chr21', 10001128, '39M61H', -409], ['chr21', 10001128, '39M61H', -511],
                #  ['chr21', 10001131, '36M64H', -498], ['chr21', 10001134, '33M67H', -557]]
                # this is 1
                # big
                # range:
                # [['chr21', 15000165, '64H36M', 441], ['chr21', 15000165, '42S58M', 490],
                #  ['chr21', 15000165, '33S67M', 448], ['chr21', 15000165, '48S52M', 441],
                #  ['chr21', 15000165, '42S58M', 439], ['chr21', 15000165, '30S70M', 1521],
                #  ['chr21', 15000165, '35S65M', 1439], ['chr21', 15000165, '49S51M', 1513],
                #  ['chr21', 15000165, '34S66M', 1586], ['chr21', 15000165, '33S67M', 1523],
                #  ['chr21', 15000165, '58H42M', 1503], ['chr21', 15002102, '65M35S', -1498],
                #  ['chr21', 15002108, '59M41S', -459], ['chr21', 15002117, '50M50H', -440],
                #  ['chr21', 15002118, '49M51H', -1514], ['chr21', 15002124, '43M57H', -1471],
                #  ['chr21', 15002124, '43M57H', -1522], ['chr21', 15002131, '36M64H', -500],
                #  ['chr21', 15002132, '35M65H', -352], ['chr21', 15002133, '34M66H', -1522],
                #  ['chr21', 15002133, '34M66H', -414], ['chr21', 15002136, '31M69H', -413]]
                # this is 2
                # big
                # range:
                if (pos == 4 or pos == 5) and bool(not100Mperrange[i][j][2][pos+1:]) == False:
                    # 如果start为0
                    if not start:
                        start = not100Mperrange[i][j][1]
                elif (pos == 4 or pos == 5) and bool(not100Mperrange[i][j][2][pos+1:]) == True:
                    flag =1
                else:
                    if pos == 1 or pos == 2:
                        end = not100Mperrange[i][j][1] + (int)(not100Mperrange[i][j][2][0: pos]) - 1

            # if (start == 0 and end == 0):
            #     # continue
            #     start = cnv_range[i][0]
            #     end = cnv_range[i][1]
            #     # flag = 0
            if (start == 0):
                start = cnv_range[i][0]

            if (end == 0):
                end = cnv_range[i][1]

            if (start > end):
                temp = start
                start = end
                end = temp

            if (end - start > 2 * binSize):
                bp_exact_position.append([int(start),int(end),flag])

    return bp_exact_position


def prox_L1(step_size: float, x: np.ndarray) -> np.ndarray:
    """
    L1 proximal operator
    """
    return np.fmax(x - step_size, 0) - np.fmax(- x - step_size, 0)

def prox_tv1d(step_size: float, w: np.ndarray) -> np.ndarray:
    """
    Computes the proximal operator of the 1-dimensional total variation operator.

    This solves a problem of the form

         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2

    where TV(x) is the one-dimensional total variation

    Parameters
    ----------
    w: array
        vector of coefficients
    step_size: float
        step size (sometimes denoted gamma) in proximal objective function

    References
    ----------
    Condat, Laurent. "A direct algorithm for 1D total variation denoising."
    IEEE Signal Processing Letters (2013)
    """

    if w.dtype not in (np.float32, np.float64):
        raise ValueError('argument w must be array of floats')
    w = w.copy()
    output = np.empty_like(w)
    _prox_tv1d(step_size, w, output)
    return output

@njit
def _prox_tv1d(step_size, input, output):
    """low level function call, no checks are performed"""
    width = input.size + 1
    index_low = np.zeros(width, dtype=np.int32)
    slope_low = np.zeros(width, dtype=input.dtype)
    index_up  = np.zeros(width, dtype=np.int32)
    slope_up  = np.zeros(width, dtype=input.dtype)
    index     = np.zeros(width, dtype=np.int32)
    z         = np.zeros(width, dtype=input.dtype)
    y_low     = np.empty(width, dtype=input.dtype)
    y_up      = np.empty(width, dtype=input.dtype)
    s_low, c_low, s_up, c_up, c = 0, 0, 0, 0, 0
    y_low[0] = y_up[0] = 0
    y_low[1] = input[0] - step_size
    y_up[1] = input[0] + step_size
    incr = 1

    for i in range(2, width):
        y_low[i] = y_low[i-1] + input[(i - 1) * incr]
        y_up[i] = y_up[i-1] + input[(i - 1) * incr]

    y_low[width-1] += step_size
    y_up[width-1] -= step_size
    slope_low[0] = np.inf
    slope_up[0] = -np.inf
    z[0] = y_low[0]

    for i in range(1, width):
        c_low += 1
        c_up += 1
        index_low[c_low] = index_up[c_up] = i
        slope_low[c_low] = y_low[i]-y_low[i-1]
        while (c_low > s_low+1) and (slope_low[max(s_low, c_low-1)] <= slope_low[c_low]):
            c_low -= 1
            index_low[c_low] = i
            if c_low > s_low+1:
                slope_low[c_low] = (y_low[i]-y_low[index_low[c_low-1]]) / (i-index_low[c_low-1])
            else:
                slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])

        slope_up[c_up] = y_up[i]-y_up[i-1]
        while (c_up > s_up+1) and (slope_up[max(c_up-1, s_up)] >= slope_up[c_up]):
            c_up -= 1
            index_up[c_up] = i
            if c_up > s_up + 1:
                slope_up[c_up] = (y_up[i]-y_up[index_up[c_up-1]]) / (i-index_up[c_up-1])
            else:
                slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

        while (c_low == s_low+1) and (c_up > s_up+1) and (slope_low[c_low] >= slope_up[s_up+1]):
            c += 1
            s_up += 1
            index[c] = index_up[s_up]
            z[c] = y_up[index[c]]
            index_low[s_low] = index[c]
            slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])
        while (c_up == s_up+1) and (c_low>s_low+1) and (slope_up[c_up]<=slope_low[s_low+1]):
            c += 1
            s_low += 1
            index[c] = index_low[s_low]
            z[c] = y_low[index[c]]
            index_up[s_up] = index[c]
            slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

    for i in range(1, c_low - s_low + 1):
        index[c+i] = index_low[s_low+i]
        z[c+i] = y_low[index[c+i]]
    c = c + c_low-s_low
    j, i = 0, 1
    while i <= c:
        a = (z[i]-z[i-1]) / (index[i]-index[i-1])
        while j < index[i]:
            output[j * incr] = a
            output[j * incr] = a
            j += 1
        i += 1
    return

@njit
def prox_tv1d_cols(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along columns of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_cols):
        _prox_tv1d(stepsize, A[:, i], out[:, i])
    return out.ravel()

@njit
def prox_tv1d_rows(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along rows of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_rows):
        _prox_tv1d(stepsize, A[i, :], out[i, :])
    return out.ravel()

def boxplot(scores,percent):
    four = pd.Series(scores).describe()
    Q1 = four['25%']
    Q3 = four['75%']
    IQR = Q3 - Q1
    upper = Q3 + percent * IQR
    lower = Q1 - percent * IQR
    print(Q1,Q3)
    return lower
# get params

start = time.time()
# purity = sys.argv[3]

reference = sys.argv[1]
bam = sys.argv[2]
splitters_bam = sys.argv[3]
discordants_bam = sys.argv[4]
purity = bam.split('_')[1]
purity = float(purity)
# depth = 8

# num = 1





# threshold = 0.25
# chrName = "human_g1k_v37.fasta"
# bam = f"/media/jhua/新加卷/19239/19239chr20_sorted.bam"
# splitters_bam = f'/media/jhua/新加卷/19239/19239chr20.splitters_sorted.bam'
# discordants_bam = '/home/jhua/桌面/simu/1_4x_0.6/mycnv4_0.6_1discordants_sorted.bam'

# bam = f"/media/jhua/4490505890505312/simu/mycnv{depth}_{purity}_{num}sorted.bam"
# splitters_bam = f"/media/jhua/4490505890505312/simu/mycnv{depth}_{purity}_{num}splitters_sorted.bam"
# discordants_bam = f"/media/jhua/4490505890505312/simu/mycnv{depth}_{purity}_{num}discordants_sorted.bam"

# f = bam.split('chr')[0]
# path = '/media/jhua/新加卷/simu/simudata/'
# bam = path + bam
# splitters_bam = path + splitters_bam
# discordants_bam = path + discordants_bam
# /media/jhua/新加卷/simu/simudata/mycnv20_0.4_1sorted.bam
binSize = 500
outputFile = bam.split('/')[-1] + ".txt"
# reference = "chr21.fa"
# reference = sys.argv[4]
# reference = "/media/yuantt/USB移动磁盘/realdata/reference/hg38/" + reference
# chrList = get_chrlist(bam)
chrList = get_chrlist(bam)
chrNum = len(chrList)
refList = [[] for i in range(chrNum)]
# chrList = []
# chr = reference.split('/')[-1].strip('chr').split('.')[0]
# # print(chr)
# chrList.append(int(chr))

# print(chrList == int(chr))
# print(np.argwhere(chrList == int(chr)))

# print(chrList)
# print(chrList) #[1 2 3 ... 0 0 0]
# print(len(chrList)) #3366
# chrNum = len(chrList)
# chrNum = 1
# refList = [[] for i in range(chrNum)]
maxbin = 3
str1 = '100M'
# reference = "chr21_19.fa"

alpha = 0.25

# percent = 0.25
cnv_range = []
# 获取参考序列
for i in range(chrNum):
    refList = read_ref_file(reference, refList, i)
seqlen = len(refList)
chrLen = np.full(chrNum, 0)
modeList = np.full(chrNum, 0.0)
for i in range(chrNum):
    chrLen[i] = len(refList[i])
print("Read bam file:", bam)
#原程序
ReadCount = np.full((chrNum, np.max(chrLen)), 0)
# ReadCount = get_RC(bam, chrList, ReadCount,chr)
mapq = np.full((chrNum, np.max(chrLen)), 0.0)
# mapq = get_mapq(bam, chrList, mapq,chr)
ReadCount, mapq = get_RC(bam, chrList, ReadCount, mapq)
for i in range(chrNum):
    binNum = int(chrLen[i]/binSize)+1
    pos, RD, MQ ,GC = ReadDepth(mapq[0], ReadCount[0], binNum, refList[i],binSize)
    for m in range(len(RD)):
        if np.isnan(RD[m]).any():   #如果RD[m]中所有点的RC都是ＮaN，则用前一个bin和后一个bin的平均值
            RD[m] = (RD[m-1] + RD[m+1]) / 2
    modeList[i] = modeRD(RD)   # 找出最密集的RD值
    # print("modeList[i]:" + modeList)
    RD = scaling_RD(RD, modeList[i])
    MQ = MQ/RD
    # modeList[i] = modeRD(RD)
# print(MQ)
print(binNum)

# test
# RD = np.loadtxt(open(r'chr21tdtest1_20x_0.4_sorted.bam.csv'))
# pos = np.loadtxt(open(r'chr21tdtest1_20x_0.4_sorted.bam_pos.csv'))
# MQ = np.loadtxt(open(r'chr21tdtest1_20x_0.4_sorted.bam_mq.csv'))
#pos代表的是一个一个碱基,不是类似于索引一样的位置
pos = pos/binSize #[ 9.413  9.414  9.415 ... 48.117 48.118 48.119]
# print(pos)
# MQ = MQ/RD

# np.savetxt('chr21tdtest1_20x_0.4_sorted.bam.csv',RD,fmt='%f')
# np.savetxt('chr21tdtest1_20x_0.4_sorted.bam_pos.csv',pos,fmt='%f')
# np.savetxt('chr21tdtest1_20x_0.4_sorted.bam_mq.csv',MQ,fmt='%f')
RDMQPOS = np.full((len(RD), 2), 0.0)
RDMQPOS[:, 0] = RD
# RDMQPOS[:, 1] = pos
RDMQPOS[:, 1] = MQ
# RDMQPOS[:, 2] = pos
# RDMQPOS[:, 2] = MQ
res = prox_tv1d(alpha, RDMQPOS[:, 0])
RDMQPOS[:, 0] = res
res = prox_tv1d(alpha, RDMQPOS[:, 1])
RDMQPOS[:, 1] = res
# res = prox_tv1d(alpha, RDMQPOS[:, 2])
# RDMQPOS[:, 2] = res
data = RDMQPOS #用tv模型对这两个特征进行平滑去噪

# data = [list(item) for item in zip(RD,MQ,pos)]
# data = np.array(data,dtype=np.float32)#把列表转化为数组形式
Data = StandardScaler().fit_transform(data)#对数据进行无量钢化
data = pd.DataFrame(Data)
# print('data')
# print(data[1])

# 训练一个OCSVM检测器
clf_name = 'OCSVM'
con = 0.15
# con = sys.argv[1]
clf = OCSVM(cache_size=6000,contamination=con)#初始化检测器clf
clf.fit(data)#使用data训练检测器clf
data_pred = clf.labels_  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值),返回异常值的索引,然后根据索引得到变异点的位置和rd
index = data_pred < 1
data_scores = clf.decision_scores_  # 返回训练数据上的异常值 (分值越大越异常)
labels_ =[]
scores = data_norm(data_scores)#将scores归一化
# print('scores类型:',type(scores[0]))
#print(scores)[0.4859267  0.33068073 0.33062829 ... 0.20563057 0.27135093 0.20570884]
rough_cnv_range = merge_bin(data_pred, maxbin, pos,binSize,RD,MQ,scores)
scores_avg = []
# rd_avg = []
rough_cnv_range_remove = []
for line in rough_cnv_range:
    scores_avg.append(line[-1])
    # rd_avg.append(line[2])
# rd_avg = np.array(rd_avg)
scores_avg = np.array(scores_avg)
# print(scores_avg)
# print('scores_avg:',type(scores_avg[0]))
# s,e,r,sc = return_num(rough_cnv_range)
#以写模式打开`test.csv`
# mkdir = bam.split('/')[-2]
nosr_name = f'/home/jhua/试验结果1/rdmq/con={con}_rough_cnv_range' + str(binSize) + bam.split('/')[-1] + '.csv'
np.savetxt(nosr_name,rough_cnv_range,fmt='%f')


# lower = np.quantile(scores_avg, 0.25, interpolation='lower')
# higher = np.quantile(scores_avg, 0.75, interpolation='higher')
# lower = np.quantile(scores, 0.40, interpolation='lower')
# higher = np.quantile(scores, 0.85, interpolation='higher')
# lower = scores_avg.min()
# higher = scores_avg.max()
# print(lower,higher)

# threshold = choice_threshold(scores_avg,lower,higher)
# print('threshold:',threshold)
# index = scores_avg <= threshold
RD_avg = np.mean(RD[index == True])# 移除变异区域的RD均值,这个可用作正常拷贝数的基线估计值
print('length of RD_avg:',RD_avg)
#
# for cnv in rough_cnv_range:
#     if cnv[-1] > threshold and cnv[-2] > 20:
#         rough_cnv_range_remove.append(cnv)
# # remove_by_threshold = f'/home/jhua/试验结果realdata/threshold/rough_cnv_range_remove' + str(binSize) + bam.split('/')[-1] + '.csv'
# # np.savetxt(remove_by_threshold,rough_cnv_range_remove,fmt='%f')
#


# RD_avg = np.mean(RD[index == True])# 移除变异区域的RD均值,这个可用作正常拷贝数的基线估计值
# print('length of RD_avg:',RD_avg)




RD_avg_up = RD_avg * (1 + purity/2)
RD_avg_down = RD_avg * (1 - purity/2)
# RD_avg_up = RD_avg * 1.1
# RD_avg_down = RD_avg * 0.9
print('RD_avg_up:',RD_avg_up)
print('RD_avg_down:',RD_avg_down)
for line in rough_cnv_range:
    line_start,line_end,line_rd,line_mq,line_score = line
    if line_rd >= RD_avg_up:
        cnv_range.append(line)
    if line_rd <= RD_avg_down:
        cnv_range.append(line)

# cnv_name = f'/home/jhua/试验结果1/rdpass/cnv_range_rdpass' + str(binSize) + bam.split('/')[-1] + '.csv'
# np.savetxt(cnv_name,cnv_range,fmt='%f')
# def get_exact_cnv_position(reference,binSize,bam,cnv_range):

# cnv_exact_range = get_exact_cnv_position(reference,binSize,discordants_bam,cnv_range)
cnv_exact_range = get_exact_cnv_position(reference,binSize,discordants_bam,cnv_range)
# print(cnv_exact_range)
bp_exact_position = get_exact_position(reference,binSize, splitters_bam, cnv_exact_range)

# 利用得到的位置,反推i的值,得到RD[i]
# gain1为穿插重复插入
with open(outputFile, 'w', encoding='utf-8') as result_file:
    result_file.write("chr" + '\t' + "start" + '\t' + "end" + '\t' + "type" + '\n')
    for i in bp_exact_position:
        cnv_start = int(i[0] // binSize)
        cnv_end = int(i[1] // binSize)
        CNV_rd = []
        # 第一,让i在len(RD)中循环,如果pos[i]*binsize in i,就返回相应的i,最后求RD均值
        for j in range(len(pos)):
            if cnv_start <= pos[j]*binSize <= cnv_end:
                CNV_rd.append(RD[j])
        # print(CNV_rd)
        CNV_rd_avg = np.mean(CNV_rd)
        # print(CNV_rd_avg)
        if (CNV_rd_avg > RD_avg) and bool(i[-1]) == True:
            result_file.write(str(reference.split('.')[0]) + '\t' +str(i[0]) + '\t' + str(i[1]) + '\t' + 'interspersed dup' + '\n')
            # print(f'{i}\t{CNV_rd_avg}\tgain')
        elif (CNV_rd_avg > RD_avg) and bool(i[-1]) == False:
            result_file.write(str(reference.split('.')[0]) + '\t' + str(i[0]) + '\t' + str(i[1]) + '\t' + 'tandem dup' + '\n')
        else:
            result_file.write(str(reference.split('.')[0]) + '\t' + str(i[0]) + '\t' + str(i[1]) + '\t' + 'loss' + '\n')
            # print(f'{i}\t{CNV_rd_avg}\tloss')
        ###333
        # if CNV_rd_avg > RD_avg:
        #     result_file.write(str(reference.split('.')[0]) + '\t' +str(i[0]) + '\t' + str(i[1]) + '\t' + 'gain' + '\n')
        #     # print(f'{i}\t{CNV_rd_avg}\tgain')
        # else:
        #     result_file.write(str(reference.split('.')[0]) + '\t' + str(i[0]) + '\t' + str(i[1]) + '\t' + 'loss' + '\n')
        #     # print(f'{i}\t{CNV_rd_avg}\tloss')
end = time.time()
print(" ** the run time of is: ", end-start, " **")


