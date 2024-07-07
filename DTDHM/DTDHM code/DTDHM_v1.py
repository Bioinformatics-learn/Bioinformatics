# -*-coding:gb2312-*-
import math
import shutil
import numpy as np
import pysam
import sys
import pandas as pd
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
from numba import njit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter
import subprocess
import time
import os
import re


def read_ref_file(filename, ref):
    # read reference file
    # input: fasta file and ref array.
    # output: ref array
    if os.path.exists(filename):
        print("Read reference file: " + str(chrname))
        with open(filename, 'r') as f:
            line = f.readline()
            for line in f:
                linestr = line.strip()
                ref += linestr
    else:
        print("Warning: can not open " + str(filename) + '\n')
    return ref


def get_RCmapq(filename, ReadCount, mapq):
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
    # get read depth
    '''
       1. compute the mean of rc in each bin;
       2. count the number of 'N' in ref. If there is a 'N' in a bin£¬the rd is not counted;
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
    # res = prox_tv1d(alpha, GC)
    # GC = res
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


def Read_seg_file():
    seg_start = []
    seg_end = []
    seg_count = []
    seg_len = []
    with open(newsegname, 'r') as f:
        for line in f:
            linestrlist = line.strip().split('\t')
            if len(linestrlist) > 5:
                start = int(linestrlist[2]) - 1
                end = int(linestrlist[3]) - 1
                seg_start.append(start)
                seg_end.append(end)
                seg_len.append(int(linestrlist[4]))
                seg_count.append(float(linestrlist[5]))
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
    return seg_RD, seg_MQ, seg_Start, seg_End


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
            type.append("gain")
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
    # discordant file
    discordantrange = np.empty(len(SVstart), dtype=object)
    dbf = pysam.AlignmentFile(drbam, 'rb')
    for i in range(len(SVstart)):
        discordantrange[i] = []
        SVlen[i] = abs(SVend[i] - SVstart[i])
        for r in dbf.fetch(chrname, SVstart[i] - maxbin * binSize,
                          SVend[i] + maxbin * binSize):
            if r.tlen != 0:
                discordantrange[i].append([r.reference_name, r.pos, r.cigarstring, r.pnext, r.tlen, SVlen[i], r.flag])
    with open(discordantresult, 'w') as f1:
        for i in range(len(discordantrange)):
            f1.write("\nthis is " + str(i) + " discordant range:\n")
            f1.writelines(str(discordantrange[i]))
    # split read file
    not100Mperrange = np.empty(len(SVstart), dtype=object)
    bf = pysam.AlignmentFile(bam, 'rb')
    for i in range(len(SVstart)):
        not100Mperrange[i] = []
        SVlen[i] = abs(SVend[i] - SVstart[i])
        for r in bf.fetch(chrname, SVstart[i] - maxbin * binSize,
                          SVend[i] + maxbin * binSize):
            if r.cigarstring != str1 and r.cigarstring != str0 and r.cigarstring != str2 \
                    and r.cigarstring != None and r.tlen != 0:
                not100Mperrange[i].append([r.reference_name, r.pos, r.cigarstring, r.pnext, r.tlen, SVlen[i],
                                           r.flag & 64, r.flag & 128, r.flag, r.query_name])
    with open(cigarresult, 'w') as f1:
        for i in range(len(not100Mperrange)):
            f1.write("\nthis is " + str(i) + " big range:\n")
            f1.writelines(str(not100Mperrange[i]))
    startlist = np.empty(len(SVstart), dtype=object)
    endlist = np.empty(len(SVstart), dtype=object)
    s_rname = np.empty(len(SVstart), dtype=object)
    e_rname = np.empty(len(SVstart), dtype=object)
    type_flag = np.full(len(SVstart), 0)
    start = np.full(len(SVstart), 0)
    end = np.full(len(SVstart), 0)
    for i in range(len(not100Mperrange)):
        startlist[i] = []
        endlist[i] = []
        s_rname[i] = []
        e_rname[i] = []
        for j in range(len(not100Mperrange[i])):
            if 'M' in not100Mperrange[i][j][2]:
                pos = not100Mperrange[i][j][2].index('M')
                if pos == 4 or pos == 5 or pos == 6:   # ySxM
                    if SVstart[i] - maxbin * binSize < not100Mperrange[i][j][1] < SVstart[i] + maxbin * binSize:
                        startlist[i].append(not100Mperrange[i][j][1])
                        s_rname[i].append(not100Mperrange[i][j][9])
                if pos == 1 or pos == 2 or pos == 3:    # xMyS
                    if SVend[i] - maxbin * binSize <= not100Mperrange[i][j][1] <= SVend[i] + maxbin * binSize:
                        if str(not100Mperrange[i][j][2][0: pos]).isdigit():
                            endlist[i].append(not100Mperrange[i][j][1] + int(not100Mperrange[i][j][2][0: pos]) - 1)
                            e_rname[i].append(not100Mperrange[i][j][9])
        # print("startlist:", startlist)
        # print("endlist:", endlist)
        if len(startlist[i]) != 0:
            start[i] = max(set(startlist[i]), key=startlist[i].count)
            SVlen[i] = SVend[i] - start[i]
        if len(endlist[i]) != 0:
            end[i] = max(set(endlist[i]), key=endlist[i].count)
            SVlen[i] = end[i] - SVstart[i]
        if len(startlist[i]) != 0 and len(endlist[i]) != 0:
            SVlen[i] = end[i] - start[i]

        dr_pos = []
        dr_pnext = []
        for l in range(len(discordantrange[i])):
            if abs(SVlen[i] - abs(discordantrange[i][l][4])) < (maxbin+2)*binSize:
                if discordantrange[i][l][4] > 0 and \
                        (abs(discordantrange[i][l][1] - SVstart[i]) < maxbin*binSize):
                    dr_pos.append(discordantrange[i][l][1])
                    dr_pnext.append(discordantrange[i][l][3])
                if discordantrange[i][l][4] < 0 and (abs(discordantrange[i][l][1] - SVend[i]) < maxbin*binSize):
                    dr_pos.append(discordantrange[i][l][3])
                    dr_pnext.append(discordantrange[i][l][1])
        if start[i] == 0:
            if dr_pos:
                start[i] = dr_pos[0]
        if end[i] == 0:
            if dr_pnext:
                end[i] = dr_pnext[-1] + readlgth
        if startlist[i]:
            td_flag1 = []
            m = int(find_first_index(startlist[i]))
            read4c = []
            for rn in bf.fetch(chrname, SVend[i] - binSize, SVend[i] + (maxbin+2) * binSize):
                if rn.query_name == s_rname[i][m] and abs(rn.pos - startlist[i][m]) > 3:
                    if 'M' in rn.cigarstring:
                        rn_Mpos = rn.cigarstring.index('M')
                        if rn_Mpos == 1 or rn_Mpos == 2 or rn_Mpos == 3:  # MS
                            if str(rn.cigarstring[0: rn_Mpos]).isdigit():
                                rn_breakp = int(rn.pos) + int(rn.cigarstring[0: rn_Mpos])
                                # find other_pos=s_brek_pos,if other_pos's cigar all MS (TD),if has SM (ISP)
                                for rm in bf.fetch(chrname, rn_breakp - 100, rn_breakp + 100):
                                    if rn.query_name != rm.query_name and rm.cigarstring != str1 and rm.cigarstring != str0 and rm.cigarstring != str2:
                                        if rm.cigarstring is not None and 'M' in rm.cigarstring:
                                            rm_Mpos = rm.cigarstring.index('M')
                                            if str(rm.cigarstring[0: rm_Mpos]).isdigit() and \
                                                    (rm_Mpos == 1 or rm_Mpos == 2 or rm_Mpos == 3):
                                                q = abs(int(rm.pos) + int(rm.cigarstring[0: rm_Mpos]) - rn_breakp)
                                                if q <= 2:
                                                    td_flag1.append('td')
                                            elif (abs(rm.pos - rn_breakp) <= 2) and \
                                                    (rm_Mpos == 4 or rm_Mpos == 5 or rm_Mpos == 6):   # SM
                                                read4c.append([rm.query_name, rm.pos, rm.cigarstring])
            for l in range(len(read4c)):
                for rm2 in bf.fetch(chrname,SVend[i]-2*binSize,read4c[l][1]):
                    if rm2.query_name == read4c[l][0] and rm2.pos != read4c[l][1] and rm2.cigarstring is not None and 'M' in rm2.cigarstring and rm2.cigarstring != str1:
                        rm2_Mpos = rm2.cigarstring.index('M')
                        if rm2_Mpos == 1 or rm2_Mpos == 2 or rm2_Mpos == 3:
                            td_flag1.append('isp')
            if td_flag1 and all(element == 'td' for element in td_flag1):
                type_flag[i] = 1  # TD
            elif 'isp' in td_flag1:
                type_flag[i] = 2  # ISP
        if endlist[i]:
            td_flag2 = []
            e = int(find_first_index(endlist[i]))
            read4c2 = []
            for rp in bf.fetch(chrname, endlist[i][e] - 100, endlist[i][e]+100):
                if rp.cigarstring is not None and 'M' in rp.cigarstring and rp.query_name != e_rname[i][
                    e] and rp.cigarstring != str1 \
                        and rp.cigarstring != str0 and rp.cigarstring != str2:
                    rp_Mpos = rp.cigarstring.index('M')
                    if abs(rp.pos - endlist[i][e]) <= 2 and (rp_Mpos == 4 or rp_Mpos == 5 or rp_Mpos == 6):
                        read4c2.append([rp.query_name, rp.pos, rp.cigarstring])
                    elif (rp_Mpos == 1 or rp_Mpos == 2 or rp_Mpos == 3) and str(
                            rp.cigarstring[0: rp_Mpos]).isdigit():  # MS
                        rp_breakpos = int(rp.pos) + int(rp.cigarstring[0: rp_Mpos])
                        q = abs(rp_breakpos - endlist[i][e])
                        if 0 <= q <= 2:
                            td_flag2.append('td')
            for t in range(len(read4c2)):
                for rp2 in bf.fetch(chrname, read4c2[t][1] - 100, read4c2[t][1]):
                    if rp2.query_name == read4c2[t][0] and rp2.pos != read4c2[t][1] and rp2.cigarstring != str1 and rp2.cigarstring is not None and 'M' in rp2.cigarstring:
                        rp2_Mpos = rp2.cigarstring.index('M')
                        if rp2_Mpos == 1 or rp2_Mpos == 2 or rp2_Mpos == 3:  # rp2:R4_b
                            td_flag2.append('isp')
            if td_flag2 and all(element == 'td' for element in td_flag2) and type_flag[i] != 2:
                type_flag[i] = 1  # TD
            elif 'isp' in td_flag2:
                type_flag[i] = 2  # ISP
        if len(endlist[i]) != 0 and type_flag[i] == 2:
            end[i] = endlist[i][0]
        if type_flag[i] == 1:
            type[i] = "TANDUP"
        elif type_flag[i] == 2:
            type[i] = "ISP"
        elif type_flag[i] == 0:
            type[i] = "TANDUP"
        if (end[i] > start[i]) and start[i] != 0 and len(dr_pos) != 0:
            SVStart.append(start[i])
            SVEnd.append(end[i])
            SVtype.append(type[i])

    return SVStart, SVEnd, SVtype


def Write_step1_file(SV_RD, SV_MQ, SV_Start, SV_End, SV_scores):
    """
    write svdata file
    svRD, svMQ, svstart, svend, averscores
    """
    output = open(bam + '_merge_step1.csv', "w")
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
    output = open(rdknnresult, "w")
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

    output = open(KNNrange, "w")
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
    # lower = Q1 - 0.8 * IQR
    return upper


# def plotRDMQ(RD, MQ):
#     # plot RD and MQ
#     plt.scatter(RD, MQ, s=3, c="black")
#     plt.xlabel("RD")
#     plt.ylabel("MQ")
#     plt.show()
#
#
# def plotscores(scores):
#     y = np.arange(1, len(scores) + 1, 1)
#     plt.scatter(scores, y, s=3)
#     plt.xlim(0, max(scores) + min(scores))
#     plt.ylim(1, len(scores) + 1)
#     plt.xlabel("scores")
#     plt.savefig(bam + '_scores.jpg')
#     plt.show()


start = time.time()
binSize = 1000
k_percent = 0.2
box_theta = 0.6
refpath = sys.argv[1]
bam = sys.argv[2]
drbam = sys.argv[3]
str1 = sys.argv[4]
# refpath="/.../chr21.fa"
# bam = "/.../chr21_sorted.bam"
# drbam = "/.../chr21_sorted_discordants.bam"
# str1 = "100M"

readlgth = int(str1.strip('M'))   # 100M --> 100
str0 = str(readlgth-1) + "M"
str2 = str(readlgth+1) + "M"
chrname = os.path.splitext(os.path.basename(refpath))[0]  # chr21
bamname = os.path.splitext(os.path.basename(bam))[0]  # chr21_8x_0.6_sorted
chrnumb = chrname.strip('chr')  # 21

TDresult = bamname + '_result.txt'
rdknnresult = bamname + '_RD+KNN.csv'
cigarresult = bamname + "_range_ciagr.txt"
discordantresult = bamname + "_range_discordant.txt"
maxbin = 5
refList = []
refList = read_ref_file(refpath, refList)
chrLen = len(refList)
print("Read bam file:", bamname)
print("chrLen:" + str(chrLen))
ReadCount = np.full(chrLen, 0)
mapq = np.full(chrLen, 0)
ReadCount, mapq = get_RCmapq(bam, ReadCount, mapq)
binNum = int(chrLen / binSize) + 1
pos, RD, MQ, GC = ReadDepth(mapq, ReadCount, binNum, refList, binSize)
alpha = 0.15
RD, MQ = TV_smoothnoise(RD, MQ)
mode = np.mean(RD)
modeMQ = np.mean(MQ)
if math.ceil(mode) >= 8:
    box_theta = -0.5
print("modeRD:" + str(mode))
print("modeMQ:" + str(modeMQ))
RD_file = bamname + "_RD"
with open(RD_file, 'w') as file:
    for i in range(len(RD)):
        file.write(str(RD[i]) + '\n')
print("RD length:", len(RD))
current_dir = os.path.dirname(os.path.realpath(__file__))
RD_file_path = os.path.join(current_dir, RD_file)
# segment
newsegname = RD_file + "_seg"
subprocess.run(["Rscript", "CBS_data.R", RD_file_path])
seg_start, seg_end, seg_count, seg_len = Read_seg_file()
KNNrange = bamname + "_range_KNN.csv"

if len(seg_start) != 0:
    seg_RD, seg_MQ, seg_Start, seg_End = seg_RDposMQ(RD, pos, MQ, seg_start, seg_end, seg_count, binSize)
    seg_chr = []
    seg_chr.extend(chrnumb for i in range(len(seg_RD)))
    seg_chr = np.array(seg_chr)
    RDMQ = np.full((len(seg_RD), 2), 0.0)
    RDMQ[:, 0] = seg_RD
    RDMQ[:, 1] = seg_MQ
    scler = StandardScaler()
    rdmq_scaler = scler.fit_transform(RDMQ)
    RDMQ_scaler = pd.DataFrame(rdmq_scaler)
    print("CBS_range:", len(seg_RD))
    # KNN
    k_neighbors = int(len(seg_RD) * k_percent)
    print("k_neighbors:", k_neighbors)
    clf = KNN(n_neighbors=k_neighbors, method='mean')
    clf.fit(RDMQ_scaler)
    labels = clf.labels_
    scores = clf.decision_scores_
    # Write_data_file(seg_chr, seg_Start, seg_End, seg_RD, seg_MQ, scores, labels)
    threshold = boxplot(scores)
    # threshold = np.mean(scores)
    print("threshold:" + str(threshold))
    SVRD, SVMQ, SVstart, SVend, SVscores, type = filter_range(mode, threshold, seg_Start, seg_End, seg_RD, seg_MQ, scores)
    # SVRD, SVMQ, SVstart, SVend, SVscores = combiningSV(SVRD, SVMQ, SVstart, SVend, SVscores)
    # Write_step2_file(SVRD, SVMQ, SVstart, SVend, SVscores)
    SV_range = np.full((len(SVstart), 2), 0)
    SV_range[:, 0] = SVstart
    SV_range[:, 1] = SVend
    print("RD+KNN_range:" + str(len(SV_range)))
    # SR+PEM
    SVStart, SVEnd, SVtype = get_exact_position(chrname, binSize, bam, str1, SVstart, SVend, type)
    print("SR+PEM_range:" + str(len(SVStart)))
    number = 1
    output = open(TDresult, "w")
    for i in range(len(SVStart)):
            output.write(
                str(chrname) + '\t' + str(SVStart[i]) + '\t' + str(SVEnd[i]) + '\t' + str(SVtype[i]) +
                '\t' + str(SVEnd[i]-SVStart[i]+1) + '\t' + str(number) + '\t' + '\n')
            print(chrname, SVStart[i], SVEnd[i], SVtype[i], SVEnd[i]-SVStart[i]+1, number)
            number += 1

else:
    output0 = open(TDresult, "w")
    print("no result")

folder_name = "DTDHM_" + bamname
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
if os.path.exists(TDresult):
    shutil.move(TDresult, os.path.join(folder_name, TDresult))
if os.path.exists(cigarresult):
    shutil.move(cigarresult, os.path.join(folder_name, cigarresult))
if os.path.exists(discordantresult):
    shutil.move(discordantresult, os.path.join(folder_name, discordantresult))
# if os.path.exists(KNNrange):
#     shutil.move(KNNrange, os.path.join(folder_name, KNNrange))
# if os.path.exists(rdknnresult):
#     shutil.move(rdknnresult, os.path.join(folder_name, rdknnresult))
# file_path1 = "RD"
# file_path2 = "seg"
if os.path.exists(RD_file):
    os.remove(RD_file)
if os.path.exists(newsegname):
    os.remove(newsegname)
end = time.time()
print(" ** the run time of is: ", end - start, " **")
