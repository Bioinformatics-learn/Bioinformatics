# -*-coding:gb2312-*-
import numpy as np
import pysam
import sys
import pandas as pd
from pyod.models.mcd import MCD
import matplotlib.pyplot as plt
from numba import njit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import time
import os
import re
from imblearn.under_sampling import RandomUnderSampler
import subprocess
import math
import shutil
from collections import Counter
from sklearn.model_selection import GridSearchCV


def get_chrlist(filename):
    # get the list of chromosomes
    # input：bam file
    # output：the list of chromosomes 
    samfile = pysam.AlignmentFile(filename, "rb")
    List = samfile.references
    chrList = np.full(len(List), 0)
    for i in range(len(List)):
        chr = str(List[i]).strip('chr')
        if chr.isdigit():
            chrList[i] = int(chr)
    index = chrList > 0
    chrList = chrList[index]  
    return chrList 


def read_ref_file(filename, ref):
    # read reference file
    # input：fasta file, ref array and the chromosomes
    # output： ref array 
    if os.path.exists(filename):
        print("Read reference file: " + str(filename))
        with open(filename, 'r') as f:
            line = f.readline()
            for line in f:
                linestr = line.strip()
                ref += linestr
    else:
        print("Warning: can not open " + str(filename) + '\n')
    return ref


def get_RCmapq(filename, ReadCount, mapq):
    # get read count for each position
    # input：bam file, the list of chromosomes and rc array
    # output：rc array 
    samfile = pysam.AlignmentFile(filename, "rb")
    for line in samfile:
        if line.reference_name:  
            chr = line.reference_name.strip('chr')
            if chr.isdigit() and chr == chrnumb:
                posList = line.positions 
                ReadCount[posList] += 1
                mapq[posList] += line.mapq
    return ReadCount, mapq


def ReadDepth(mapq, ReadCount, binNum, ref, binSize):
    '''
       1. compute the mean of rc in each bin;计算每个 bin 中 rc 的平均值；
       2. count the number of 'N' in ref. If there is a 'N' in a bin，the rd is not counted; 计算参考中“N”的数量。 如果 bin 中有 'N'，则不计算rd
       3. GC bias  GC 偏差
    '''
    RD = np.full(binNum, 0.0)
    GC = np.full(binNum, 0)
    MQ = np.full(binNum, 0.0)
    pos = np.arange(1, binNum + 1) 
    for i in range(binNum):
        RD[i] = np.mean(ReadCount[i * binSize:(i + 1) * binSize])
        MQ[i] = np.mean(mapq[i * binSize:(i + 1) * binSize])
        cur_ref = ref[i * binSize:(i + 1) * binSize]
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
    return pos, RD, MQ, GC


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


def TV_smoothnoise(RD, MQ):
    res = prox_tv1d(alpha, RD) 
    RD = res 
    res = prox_tv1d(alpha, MQ) 
    MQ = res  
    return RD, MQ


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
    index_up = np.zeros(width, dtype=np.int32)
    slope_up = np.zeros(width, dtype=input.dtype)
    index = np.zeros(width, dtype=np.int32)
    z = np.zeros(width, dtype=input.dtype)
    y_low = np.empty(width, dtype=input.dtype)
    y_up = np.empty(width, dtype=input.dtype)
    s_low, c_low, s_up, c_up, c = 0, 0, 0, 0, 0
    y_low[0] = y_up[0] = 0
    y_low[1] = input[0] - step_size
    y_up[1] = input[0] + step_size
    incr = 1
    for i in range(2, width):
        y_low[i] = y_low[i - 1] + input[(i - 1) * incr]
        y_up[i] = y_up[i - 1] + input[(i - 1) * incr]
    y_low[width - 1] += step_size
    y_up[width - 1] -= step_size
    slope_low[0] = np.inf
    slope_up[0] = -np.inf
    z[0] = y_low[0]
    for i in range(1, width):
        c_low += 1
        c_up += 1
        index_low[c_low] = index_up[c_up] = i
        slope_low[c_low] = y_low[i] - y_low[i - 1]
        while (c_low > s_low + 1) and (slope_low[max(s_low, c_low - 1)] <= slope_low[c_low]):
            c_low -= 1
            index_low[c_low] = i
            if c_low > s_low + 1:
                slope_low[c_low] = (y_low[i] - y_low[index_low[c_low - 1]]) / (i - index_low[c_low - 1])
            else:
                slope_low[c_low] = (y_low[i] - z[c]) / (i - index[c])
        slope_up[c_up] = y_up[i] - y_up[i - 1]
        while (c_up > s_up + 1) and (slope_up[max(c_up - 1, s_up)] >= slope_up[c_up]):
            c_up -= 1
            index_up[c_up] = i
            if c_up > s_up + 1:
                slope_up[c_up] = (y_up[i] - y_up[index_up[c_up - 1]]) / (i - index_up[c_up - 1])
            else:
                slope_up[c_up] = (y_up[i] - z[c]) / (i - index[c])
        while (c_low == s_low + 1) and (c_up > s_up + 1) and (slope_low[c_low] >= slope_up[s_up + 1]):
            c += 1
            s_up += 1
            index[c] = index_up[s_up]
            z[c] = y_up[index[c]]
            index_low[s_low] = index[c]
            slope_low[c_low] = (y_low[i] - z[c]) / (i - index[c])
        while (c_up == s_up + 1) and (c_low > s_low + 1) and (slope_up[c_up] <= slope_low[s_low + 1]):
            c += 1
            s_low += 1
            index[c] = index_low[s_low]
            z[c] = y_low[index[c]]
            index_up[s_up] = index[c]
            slope_up[c_up] = (y_up[i] - z[c]) / (i - index[c])
    for i in range(1, c_low - s_low + 1):
        index[c + i] = index_low[s_low + i]
        z[c + i] = y_low[index[c + i]]
    c = c + c_low - s_low
    j, i = 0, 1
    while i <= c:
        a = (z[i] - z[i - 1]) / (index[i] - index[i - 1])
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


def modeRD(RD):
    newRD = np.full(len(RD), 0)
    for i in range(len(RD)): 
        newRD[i] = int(round(RD[i], 3) * 1000)
    count = np.bincount(newRD) 
    countList = np.full(len(count) - 49, 0)  
    for i in range(len(countList)):
        countList[i] = np.mean(count[i:i + 50]) 
    modemin = np.argmax(countList)  
    modemax = modemin + 50
    mode = (modemax + modemin) / 2 
    mode = mode / 1000  
    return mode


def scaling_RD(RD, mode): 
    posiRD = RD[RD > mode]  
    negeRD = RD[RD < mode]
    if len(posiRD) < 50:  
        mean_max_RD = np.mean(posiRD) 
    else:
        sort = np.argsort(posiRD)  
        maxRD = posiRD[sort[-50:]] 
        mean_max_RD = np.mean(maxRD) 
    if len(negeRD) < 50:  
        mean_min_RD = np.mean(negeRD)
    else:
        sort = np.argsort(negeRD)
        minRD = negeRD[sort[:50]] 
        mean_min_RD = np.mean(minRD)
    scaling = mean_max_RD / (mode + mode - mean_min_RD) 
    for i in range(len(RD)):
        if RD[i] < mode:
            RD[i] /= scaling  
    print('RD:', RD)
    return RD


def Read_seg_file(): 
    """
    read segment file (Generated by DNAcopy.segment)
    seg file: col, chr, start, end, num_mark, seg_mean
    """
    seg_start = [] 
    seg_end = [] 
    seg_count = [] 
    seg_len = [] 
    with open(newsegname, 'r') as f:
        for line in f: 
            linestrlist = line.strip().split('\t')
            start = int(linestrlist[1]) - 1 
            end = int(linestrlist[2]) - 1 
            seg_start.append(start) 
            seg_end.append(end)
            seg_count.append(float(linestrlist[5]))
            seg_len.append(int(linestrlist[4]))
    seg_start = np.array(seg_start)  
    seg_end = np.array(seg_end)
    seg_count = np.array(seg_count) 
    return seg_start, seg_end, seg_count, seg_len


def seg_RDposMQ(RD, binHead, MQ, seg_start, seg_end, seg_count, binSize):
    seg_RD = np.full(len(seg_count), 0.0) 
    seg_MQ = np.full(len(seg_count), 0.0)
    seg_Start = np.full(len(seg_count), 0)
    seg_End = np.full(len(seg_count), 0)
    for i in range(len(seg_RD)):
        seg_RD[i] = np.mean(RD[seg_start[i]:seg_end[i]])
        seg_MQ[i] = np.mean(MQ[seg_start[i]:seg_end[i]])
        seg_Start[i] = binHead[seg_start[i]] * binSize + 1
        if seg_end[i] == len(binHead):
            seg_end[i] = len(binHead) - 1
        seg_End[i] = binHead[seg_end[i]] * binSize + binSize 
    return seg_RD, seg_Start, seg_End, seg_MQ


def MinMaxNormalization(data_list):
    scores_normal = []
    Max = np.max(data_list)
    Min = np.min(data_list)
    for data in data_list:
        new_data = (data - Min) / (Max - Min)  
        scores_normal.append(new_data) 
    return scores_normal


def merge_bin(RD, MQ, labels, maxbin, pos, binSize, scores):
    start = ends = step = 0
    svstart = []
    svend = []
    svRD = []
    svMQ = []
    averscores = []
    m = 0  
    n = 0 
    for i in range(len(labels)): 
        if labels[i] == 1 and start == 0:
            start = pos[i]
            m = i
            continue
        if labels[i] == 1 and start != 0:
            ends = pos[i]
            n = i
            step = 0
            continue
        if labels[i] != 1 and start != 0:
            if i == len(labels) - 1:  
                svstart.append(start * binSize) 
                svend.append(ends * binSize)
                svRD.append(np.mean(RD[m:n]))
                svMQ.append(np.mean(MQ[m:n]))
                averscores.append(np.mean(scores[m:n]))
            step = step + 1 
            if step == maxbin: 
                if ends - start >= maxbin:  
                    svstart.append(start * binSize) 
                    svend.append(ends * binSize)
                    svRD.append(np.mean(RD[m:n]))
                    svMQ.append(np.mean(MQ[m:n]))
                    averscores.append(np.mean(scores[m:n]))
                start = 0  
                ends = 0
                step = 0
    return svstart, svend, svRD, svMQ, averscores


def filter_range(mode, threshold, svstart, svend, svRD, svMQ, averscores):
    SVRD = []
    SVMQ = []
    SVstart = []
    SVend = []
    SVscores = []
    type = []  
    for i in range(len(svRD)):  
        if svRD[i] > mode and svMQ[i] > modeMQ and averscores[i] > threshold and svend[i] > svstart[i]:
            SVRD.append(svRD[i]) 
            SVMQ.append(svMQ[i])
            SVstart.append(svstart[i])
            SVend.append(svend[i])
            SVscores.append(averscores[i])
            type.append("TANDUP")  
    return SVRD, SVMQ, SVstart, SVend, SVscores, type


def combiningSV(SVRD, SVMQ, SVstart, SVend, SVscores):
    SV_Start = []
    SV_End = []
    SV_RD = []
    SV_MQ = []
    SV_scores = []
    SVtype = np.full(len(SVRD), 1)
    for i in range(len(SVRD) - 1):
        if SVend[i] + 1 == SVstart[i + 1]:
            SVstart[i + 1] = SVstart[i]
            SVRD[i + 1] = np.mean(SVRD[i:i + 1])
            SVMQ[i + 1] = np.mean(SVMQ[i:i + 1])
            SVscores[i + 1] = np.mean(SVscores[i:i + 1])
            SVtype[i] = 0
    for i in range(len(SVRD)):
        if SVtype[i] == 1:
            SV_Start.append(SVstart[i])
            SV_End.append(SVend[i])
            SV_RD.append(SVRD[i])
            SV_MQ.append(SVMQ[i])
            SV_scores.append(SVscores[i])
    return SV_RD, SV_MQ, SV_Start, SV_End, SV_scores


def find_first_index(arr):
    counts = Counter(arr)
    mostcommon_element = max(counts, key=counts.get)
    first_index = next(i for i, elem in enumerate(arr) if elem == mostcommon_element)
    return first_index


def get_exact_position(chrname, binSize, bam, str1, SVstart, SVend, type):
    SVStart = []  
    SVEnd = []
    SVlen = np.full(len(SVstart), 0)
    SVtype = []
    discordantrange = np.empty(len(SVstart), dtype=object)
    dbf = pysam.AlignmentFile(drbam, 'rb') 
    for i in range(len(SVstart)):
        discordantrange[i] = []  
        SVlen[i] = abs(SVend[i] - SVstart[i]) 
        for r in dbf.fetch(chrname, SVstart[i] - maxbin * binSize, SVend[i] + maxbin * binSize): 
            if r.tlen != 0:  
                discordantrange[i].append([r.reference_name, r.pos, r.cigarstring, r.pnext, r.tlen, SVlen[i], r.flag]) 
    discordantresult = "_range_discordant.txt"  
    with open(bam + discordantresult, 'w') as f1: 
        for i in range(len(discordantrange)):
            f1.write("\nthis is " + str(i) + " discordant range:\n")
            f1.writelines(str(discordantrange[i]))
    not100Mperrange = np.empty(len(SVstart), dtype=object)  
    bf = pysam.AlignmentFile(bam, 'rb')  
    for i in range(len(SVstart)):
        not100Mperrange[i] = []
        SVlen[i] = abs(SVend[i] - SVstart[i])
        if SVstart[i] - maxbin * binSize > SVend[i] + maxbin * binSize:
            continue
        for r in bf.fetch(chrname, SVstart[i] - maxbin * binSize, SVend[i] + maxbin * binSize):  
            if r.cigarstring != str1 and r.cigarstring != None and r.tlen != 0: 
                not100Mperrange[i].append([r.reference_name, r.pos, r.cigarstring, r.pnext, r.tlen, SVlen[i], r.flag & 64, r.flag & 128, r.flag])
    cigarresult = "_range_ciagr.txt"  
    with open(bam + cigarresult, 'w') as f1:
        for i in range(len(not100Mperrange)): 
            f1.write("\nthis is " + str(i) + " big range:\n")
            f1.writelines(str(not100Mperrange[i]))
            start = 0
            end = 0
            startlist = []  
            endlist = [] 
            for j in range(len(not100Mperrange[i])):  
                pos = not100Mperrange[i][j][2].index('M')
                if pos == 4 or pos == 5 or pos == 6:  
                    startlist.append(not100Mperrange[i][j][1])
                if pos == 1 or pos == 2 or pos == 3:
                    if not100Mperrange[i][j][2][0: pos].isdigit():
                        endlist.append(not100Mperrange[i][j][1] + (int)(not100Mperrange[i][j][2][0: pos]) - 1)
            if len(startlist) != 0:
                start = max(set(startlist), key=startlist.count)
                SVlen[i] = SVend[i] - start
            if len(endlist) != 0:
                end = max(set(endlist), key=endlist.count)
                SVlen[i] = end - SVstart[i]
            if len(startlist) != 0 and len(endlist) != 0:
                SVlen[i] = end - start
            # PEM
            dr_pos = []
            dr_pnext = []
            for l in range(len(discordantrange[i])): 
                if start != 0 and abs(discordantrange[i][l][1] - start) < 2 * binSize:
                    dr_pos.append(start)
                if end != 0 and abs(discordantrange[i][l][3] - end) < 2 * binSize:
                    dr_pnext.append(end)
                if abs(SVlen[i] - abs(discordantrange[i][l][4])) < 0.45 * SVlen[i] \
                        and abs(discordantrange[i][l][1] - SVstart[i]) < maxbin * binSize:
                    dr_pos.append(discordantrange[i][l][1])
                    dr_pnext.append(discordantrange[i][l][3])
            if start == 0:
                if dr_pos:
                    start = dr_pos[0]  
            if end == 0:
                if dr_pnext:
                    end = dr_pnext[-1] + 100  
            if end - start >= binSize and start != 0 and len(dr_pos) != 0:
                SVStart.append(start)
                SVEnd.append(end)
                SVtype.append(type[i])

    return SVStart, SVEnd, SVtype


def Write_step1_file(SV_RD, SV_MQ, SV_Start, SV_End, SV_scores): 
    """
    write svdata file
    svRD, svMQ, svstart, svend, averscores
    """
    output = open(bam + '_noSR_step1.txt', "w")
    output.write(
        "SV_RD" + '\t' + "SV_MQ" + '\t' + "SV_Start" + '\t' + "SV_End" + '\t' + "SV_averscores" + '\n')
    for i in range(len(SV_RD)):
        output.write(
            str(SV_RD[i]) + '\t' + str(SV_MQ[i]) + '\t' + str(SV_Start[i]) +
            '\t' + str(SV_End[i]) + '\t' + str(SV_scores[i]) + '\t' + '\n')


def Write_step2_file(SV_RD, SV_MQ, SV_Start, SV_End, SV_scores): 
    """
    write svdata file
    svRD, svMQ, svstart, svend, averscores
    """
    output = open(bam + '_filter_step2.txt', "w")
    output.write(
        "SV_RD" + '\t' + "SV_MQ" + '\t' + "SV_Start" + '\t' + "SV_End" + '\t' + "SV_scores" + '\n')
    for i in range(len(SV_RD)):
        output.write(
            str(SV_RD[i]) + '\t' + str(SV_MQ[i]) + '\t' + str(SV_Start[i]) +
            '\t' + str(SV_End[i]) + '\t' + str(SV_scores[i]) + '\n')


def Write_data_file(chr, seg_start, seg_end, seg_count, seg_mq, scores, labels):
    """
    write knn data file
    chr, start, end, rd, mq, score, label
    """
    output = open(MCDrange, "w")
    output.write(
        "chr" + '\t' + "start" + '\t' + "end" + '\t' + "readdepth" + '\t' + "mapquality" + '\t' + "score" + '\t' + "labels" +'\n')
    for i in range(len(scores)):
        output.write(
            str(chr[i]) + '\t' + str(seg_start[i]) + '\t' + str(seg_end[i]) +
            '\t' + str(seg_count[i]) + '\t' + str(seg_mq[i]) + '\t' + str(scores[i]) + '\t' + str(labels[i]) + '\n')


def boxplot(averscores): 
    four = pd.Series(averscores).describe()
    Q1 = four['25%']
    Q3 = four['75%']
    IQR = Q3 - Q1
    upper = Q3 + box_theta * IQR
    #lower = Q1 - 0.6 * IQR
    return upper


start = time.time()
binSize = 1000
box_theta = 0.3
refpath = sys.argv[1]
bam = sys.argv[2]
drbam = sys.argv[3]
str1 = sys.argv[4]

readlgth = int(str1.strip('M'))  
str0 = str(readlgth-1) + "M"
str2 = str(readlgth+1) + "M"
chrname = os.path.splitext(os.path.basename(refpath))[0] 
bamname = os.path.splitext(os.path.basename(bam))[0]
chrnumb = chrname.strip('chr') 

TDresult = bamname + '_result.txt'
rdmcdresult = bamname + '_RD+MCD.csv'
cigarresult = bamname + "_range_ciagr.txt"
discordantresult = bamname + "_range_discordant.txt"

refList = [] 
refList = read_ref_file(refpath, refList)
chrLen = len(refList) 
print("Read bam file:", bamname)
print("chrLen:" + str(chrLen))
ReadCount = np.full(chrLen, 0) 
mapq = np.full(chrLen, 0) 
ReadCount, mapq = get_RCmapq(bam, ReadCount, mapq)  
binNum = int(chrLen / binSize) + 1 
print("binNum:" + str(binNum))
pos, RD, MQ, GC = ReadDepth(mapq, ReadCount, binNum, refList, binSize) 
alpha = 0.6
RD, MQ = TV_smoothnoise(RD, MQ)
mode = modeRD(RD)
modeMQ = np.mean(MQ)
if math.ceil(mode) >= 8:
    box_theta = - 0.5
print("modeRD:" + str(mode))
print("modeMQ:" + str(modeMQ))

RD_file = bamname + "_RD"
with open(RD_file, 'w') as file:
    for i in range(len(RD)):
        file.write(str(RD[i]) + '\n')
print("RD length:", len(RD))
current_dir = os.path.dirname(os.path.realpath(__file__))
RD_file_path = os.path.join(current_dir, RD_file)
newsegname = RD_file + "_seg"
subprocess.run(["Rscript", "CBS_data.R", RD_file_path])
seg_start, seg_end, seg_count, seg_len = Read_seg_file()
MCDrange = bamname + "_range_MCD.csv"
seg_RD, seg_MQ, seg_Start, seg_End = seg_RDposMQ(RD, pos, MQ, seg_start, seg_end, seg_count, binSize)
seg_chr = []
seg_chr.extend(chrnumb for i in range(len(seg_RD)))
seg_chr = np.array(seg_chr)

RDMQ = np.full((len(RD), 2), 0.0)
RDMQ[:, 0] = RD
RDMQ[:, 1] = MQ

scler = StandardScaler()
rdmq_scaler = scler.fit_transform(RDMQ)
RDMQ_scaler = pd.DataFrame(rdmq_scaler) 

clf = MCD(support_fraction=0.7)
clf.fit(RDMQ_scaler) 
labels = clf.labels_ 

# original_scores
original_scores = clf.decision_scores_
original_scores = np.array(original_scores, dtype=float) 
# print("original_scores:", original_scores)

# normalized_scores
normalized_scores = MinMaxNormalization(original_scores)
normalized_scores = np.array(normalized_scores, dtype=float)
# print("normalized_scores:", normalized_scores)

# combined_scores
combined_scores = 0.5 * original_scores + 0.5 * normalized_scores 

maxbin = 3 
print("maxbin:", maxbin)

svstart, svend, svRD, svMQ, averscores = merge_bin(RD, MQ, labels, maxbin, pos, binSize, combined_scores) 
Write_step1_file(svRD, svMQ, svstart, svend, averscores)  

threshold = boxplot(averscores)
print("threshold:" + str(threshold))

SVRD, SVMQ, SVstart, SVend, SVscores, type = filter_range(mode, threshold, svstart, svend, svRD, svMQ, averscores)
Write_step2_file(SVRD, SVMQ, SVstart, SVend, SVscores)  

SV_range = np.full((len(SVstart), 2), 0)
SV_range[:, 0] = SVstart 
SV_range[:, 1] = SVend 
print("filter_range:" + str(len(SV_range)))

SVStart, SVEnd, SVtype = get_exact_position(chrname, binSize, bam, str1, SVstart, SVend, type)
print("SR+PEM_range:" + str(len(SVStart)))

number = 1
output = open(TDresult, "w")
for i in range(len(SVStart)):
    output.write(
        str(chrname) + '\t' + str(SVStart[i]) + '\t' + str(SVEnd[i]) + '\t' + str(SVEnd[i] - SVStart[i] + 1) + '\t' + str(number) + '\t' + '\n')
    print(chrname, SVStart[i], SVEnd[i], SVEnd[i] - SVStart[i] + 1, number)
    number += 1
folder_name = "MCD_" + bamname
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
if os.path.exists(TDresult):
    shutil.move(TDresult, os.path.join(folder_name, TDresult))
if os.path.exists(cigarresult):
    shutil.move(cigarresult, os.path.join(folder_name, cigarresult))
if os.path.exists(discordantresult):
    shutil.move(discordantresult, os.path.join(folder_name, discordantresult))

end = time.time()
print(" ** the run time of is: ", end - start, " **")
