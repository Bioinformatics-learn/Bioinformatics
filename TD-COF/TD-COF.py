#encoding: utf-8
import sys
import numpy as np
import pysam
import os
import datetime
import pandas as pd
from numba import njit
import subprocess
from pyod.models.cof import COF
from imblearn.under_sampling import RandomUnderSampler

def get_chrlist(filename):

    samfile = pysam.AlignmentFile(filename, "rb", ignore_truncation=True)
    List = samfile.references
    chrList = []
    for i in range(len(List)):
        chr = str(List[i]).strip('chr')
        if chr.isdigit():
            chrList.append(int(chr))
    return chrList

def get_RC(filename ,ReadCount,Mpq):
    breakpoint = []
    samfile = pysam.AlignmentFile(filename, "rb")
    for line in samfile:
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                posList = line.positions
                mapq = line.mapq
                ReadCount[posList] += 1
                Mpq[posList] += mapq
            if line.cigarstring != None:
                if 'S' in line.cigarstring and line.mapq > 30:
                    S_num = len(line.cigarstring.split('S')) - 1
                    H_num = len(line.cigarstring.split('H')) - 1
                    if S_num == 1:
                        S_index = line.cigarstring.index('S')
                        M_index = line.cigarstring.index('M')
                        if S_index < M_index:
                            breakpoint.append(int(line.pos))
                        elif S_index > M_index:
                            breakpoint.append(int(line.pos) + int(line.qlen))
                    elif H_num == 1:
                        H_index = line.cigarstring.index('H')
                        M_index = line.cigarstring.index('M')
                        if H_index < M_index:
                            breakpoint.append(int (line.pos))
                        elif H_index > M_index:
                            breakpoint.append(int(line.pos) + int(line.qlen))
    breakpoint.append(0)
    breakpoint.append(chrLen[20]-1)
    s = sorted(list(set(breakpoint)))
    print(s)
    return ReadCount,Mpq,s

def read_ref_file(filename, ref):
    if os.path.exists(filename):
        print("Read reference file: " + str(filename))
        with open(filename, 'r') as f:
            line = f.readline()
            chr_name = line.strip('>').strip()
            chr_num = int(line.strip('>chr'))
            for line in f:
                linestr = line.strip()
                ref[chr_num - 1] += linestr
    else:
        print("Warning: can not open " + str(filename) + '\n')
    return ref,chr_num,chr_name

def binning(binnum):
    pos = []
    for i in range(0,binnum+1):
        pos.append(i*binSize)
    return pos


def ReadDepth(ReadCount, ref, Mapq, pos):
    # get read depth and mapping qualities
    start = pos[:len(pos) - 1]
    end = pos[1:]
    for i in range(len(pos) - 1):
        if end[i] - start[i] < binSize:
            pos.remove(end[i])
    pos = np.array(pos)
    print(pos)
    start = pos[:len(pos) - 1]
    end = pos[1:]
    length = end - start
    with open('new_pos.txt', 'w') as f:
        for i in range(len(start)):
            linestrlist = ['1', '1', str(start[i]), str(end[i]-1), str(length[i])]
            f.write('\t'.join(linestrlist) + '\n')
    bin_start,bin_end,bin_len = re_segfile('new_pos.txt', 'reseg_file.txt', 1000)
    binNum = len(bin_start)
    bin_RD = np.full(binNum, 0.0)
    bin_MQ = np.full(binNum,0.0)
    bin_GC = np.full(binNum, 0)
    bin_gc = np.full(binNum,0.0)
    for i in range(binNum):
        bin_RD[i] = np.mean(ReadCount[bin_start[i]:bin_end[i]])
        bin_MQ[i] = np.mean(Mapq[bin_start[i]:bin_end[i]])
        cur_ref = ref[bin_start[i]:bin_end[i]]
        N_count = cur_ref.count('N') + cur_ref.count('n')
        if N_count == 0:
            gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
        else:
            bin_RD[i] = -10000
            gc_count = 0
        bin_GC[i] = int(round(gc_count / bin_len[i], 3) * 1000)
        bin_gc[i] = round(gc_count / bin_len[i],2)
    bin_end -= 1
    index = bin_RD > 0
    bin_RD = bin_RD[index]
    bin_MQ = bin_MQ[index]
    bin_GC = bin_GC[index]
    bin_gc = bin_gc[index]
    bin_len = bin_len[index]
    bin_start = bin_start[index]
    bin_end = bin_end[index]
    bin_RD = gc_correct(bin_RD, bin_GC)

    return bin_start,bin_end,bin_len, bin_RD, bin_MQ, bin_gc


def gc_correct(RD, GC):
    # correcting gc bias
    bincount = np.bincount(GC)
    global_rd_ave = np.mean(RD)
    global_rd_median = np.median(RD)
    for i in range(len(RD)):
        if bincount[GC[i]] < 2:
            continue
        mean_RD = np.median(RD[GC == GC[i]])
        RD[i] = global_rd_median * RD[i] / mean_RD
    return RD


def prox_tv1d(step_size, w):
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



def Read_seg_file(binstart,binlen,binend,binmq):
    """
    read segment file (Generated by DNAcopy.segment)
    seg file: col, chr, start, end, num_mark, seg_mean
    """
    seg_start = []
    seg_end = []
    seg_len = []
    location = []
    seg_pos = []
    for i in range(len(binstart) - 1):
        if binstart[i] + binlen[i] != binstart[i + 1]:
            location.append(i+1)
    print(location)
    count = 0
    with open("seg", 'r') as f1,\
            open('seg_re.txt', 'w') as f2:
        for line in f1:
            linestrlist = line.strip().split('\t')
            start = int(linestrlist[2])
            end = int(linestrlist[3])
            seg_pos.append([])
            seg_pos[-1].append(start)
            seg_pos[-1].append(end)
            for j in location:
                if j > seg_pos[-1][0] and j < seg_pos[-1][-1]:
                    seg_pos[-1].append(j)
                    seg_pos[-1].append(j + 1)
                    seg_pos[-1] = sorted(seg_pos[-1])
            for k in range(0,len(seg_pos[count]),2):
                start = seg_pos[count][k] - 1
                end = seg_pos[count][k+1]-1
                linestrlist[2] = str(binstart[seg_pos[count][k] - 1])
                seg_start.append(seg_pos[count][k]-1)
                linestrlist[3] = str(binend[seg_pos[count][k+1]-1])#
                linestrlist[4] = str(np.sum(binlen[seg_pos[count][k] - 1:seg_pos[count][k+1]]))
                seg_end.append(seg_pos[count][k+1]-1)
                seg_len.append(np.sum(binlen[start:end + 1]))
                linestrlist.append('')
                linestrlist[6] = (str(np.mean(binmq[start:end+1])))
                f2.write('\t'.join(linestrlist) + '\n')
            count += 1
    reseg_Start,reseg_End,reseg_Len = re_segfile('seg_re.txt','reseg2.txt',10000)
    reseg_Start = np.array(reseg_Start)
    reseg_End = np.array(reseg_End)
    reseg_Len = np.array(reseg_Len)
    return reseg_Start, reseg_End

def segment_RD(RD, binStart, MQ, GC, seg_start, seg_end, length):
    seg_RD = np.full(len(seg_start), 0.0)
    seg_MQ = np.full(len(seg_start), 0.0)
    seg_GC = np.full(len(seg_start), 0.0)
    for i in range(len(seg_RD)):
        seg_RD[i] = np.mean(RD[seg_start[i]:seg_end[i]+1])
        seg_MQ[i] = np.mean(MQ[seg_start[i]:seg_end[i]+1])
        seg_GC[i] = np.mean(GC[seg_start[i]:seg_end[i]+1])
        seg_start[i] = binStart[seg_start[i]] + 1
        seg_end[i] = binStart[seg_end[i]] + length[seg_end[i]]

    return seg_RD, seg_start, seg_end, seg_MQ, seg_GC

def resegment_RD(ReadCount,Mapq,start,end):
    reseg_RD = []
    reseg_MQ = []
    reseg_start = []
    reseg_end = []
    for i in range(len(start)):
        reseg_RD.append(np.mean(ReadCount[start[i]:end[i]]))
        reseg_MQ.append(np.mean(Mapq[start[i]:end[i]]))
        reseg_start.append(start[i] + 1)
        reseg_end.append(end[i])
    reseg_RD = np.array(reseg_RD)
    reseg_MQ = np.array(reseg_MQ)
    return reseg_RD, reseg_start, reseg_end, reseg_MQ

def re_segfile(filname,savefile,reseg_length):
    with open(filname,'r') as f1, \
            open(savefile,'w') as f2:
        for line in f1:
            linestrlist = line.strip().split('\t')
            start = int(linestrlist[2])
            end = int(linestrlist[3])
            length = int(linestrlist[4])
            if length > reseg_length:
                l = length % reseg_length
                if l:
                    if l >= binSize:
                        reseg_num = length // reseg_length + 1
                        for i in range(reseg_num):
                            if i == 0:
                                linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1)
                                linestrlist[4] = str(reseg_length)
                                f2.write('\t'.join(linestrlist) + '\n')
                            elif i+1 != reseg_num:
                                linestrlist[2] = str(int(linestrlist[2]) + 1 * reseg_length)
                                linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1 )
                                linestrlist[4] = str(reseg_length)
                                f2.write('\t'.join(linestrlist) + '\n')
                            else:
                                linestrlist[2] = str(int(linestrlist[2]) + 1 * reseg_length)
                                linestrlist[3] = str(int(linestrlist[2])  + l - 1)
                                linestrlist[4] = str(l)
                                f2.write('\t'.join(linestrlist) + '\n')

                    else:
                        bin_num = length // reseg_length
                        for i in range(bin_num):
                            if i == 0:
                                if i+1 != bin_num:
                                    linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1)
                                    linestrlist[4] = str(reseg_length)
                                else:
                                    linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1 + l)
                                    linestrlist[4] = str(reseg_length + l)
                                f2.write('\t'.join(linestrlist) + '\n')
                            elif i+1 != bin_num:
                                linestrlist[2] = str(int(linestrlist[2]) + 1 * reseg_length)
                                linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1 )
                                linestrlist[4] = str(reseg_length)
                                f2.write('\t'.join(linestrlist) + '\n')
                            else:
                                linestrlist[2] = str(int(linestrlist[2]) + 1 * reseg_length)
                                linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1 + l)
                                linestrlist[4] = str(reseg_length + l)
                                f2.write('\t'.join(linestrlist) + '\n')

                else:
                    bin_num = length // reseg_length
                    for i in range(bin_num):
                        if i==0:
                            linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1)
                            linestrlist[4] = str(reseg_length)
                            f2.write('\t'.join(linestrlist) + '\n')
                        else:
                            linestrlist[2] = str(int(linestrlist[2]) + 1 * reseg_length)
                            linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1)
                            linestrlist[4] = str(reseg_length)
                            f2.write('\t'.join(linestrlist) + '\n')
            else:
                f2.write('\t'.join(linestrlist) + '\n')
    tran_start = []
    tran_end = []
    tran_len = []
    with open(savefile, 'r') as f:
        for line in f:
            linestrinfo = line.strip().split('\t')
            tran_start.append(int(linestrinfo[2]))
            tran_end.append(int(linestrinfo[3]) + 1)
            tran_len.append(int(linestrinfo[4]))
    tran_start = np.array(tran_start)
    tran_end = np.array(tran_end)
    tran_len = np.array(tran_len)
    return tran_start, tran_end, tran_len

def get_newbins(new_data):
    new_chr = np.array(new_data['chr'])
    new_start = np.array(new_data['start'])
    new_end = np.array(new_data['end'])
    new_rd = np.array(new_data['rd'])
    new_mq = np.array(new_data['mq'])

    return new_chr,new_start,new_end,new_rd,new_mq

def Otsu(S):
    S = np.round(S,2)
    min_S = np.min(S)
    median_S = np.median(S)
    lower_S = np.quantile(S,0.35,method='lower')
    higer_S = np.quantile(S,0.85,method='higher')
    if(lower_S == min_S):
        lower_S += 0.1
    final_threshold = median_S
    max_var = 0.0
    D_labels = np.full(len(S), 0)
    for i in np.arange(lower_S,higer_S,0.01):
        cur_threshold = round(i,2)
        D0_index = (S < cur_threshold)
        D1_index = (S >= cur_threshold)
        D_labels[D0_index] = 0
        D_labels[D1_index] = 1
        D0 = S[D0_index]
        D1 = S[D1_index]
        S_resample = S.reshape(-1,1)
        new_D,new_label = RandomUnderSampler(random_state=42).fit_resample(S_resample,D_labels)
        new_D0 = new_D.ravel()[new_label==0]
        new_D1 = new_D.ravel()[new_label==1]

        D0_mean = np.mean(new_D0)
        D1_mean = np.mean(new_D1)
        p0 = len(D0)/(len(D0)+len(D1))
        p1 = (1 - p0)
        S_mean = p0*D0_mean + p1*D1_mean
        cur_var = p0*(D0_mean - S_mean)**2 + p1*(D1_mean - S_mean)**2
        if cur_var > max_var:
            final_threshold = cur_threshold
            max_var = cur_var
    return final_threshold

def Write_data_file(chr, start, length,  seg_count, seg_mq, scores, outfile):
    """
    write TDdata file
    """
    output = open(outfile, "w")
    output.write("chr" + '\t' + "start" + '\t' + "end" + '\t' + "length" + '\t' + "read depth" + '\t' + "mapping quality" + '\t' + "score" + '\n')
    for i in range(len(scores)):
        output.write(
            str(chr[i]) + '\t' + str(start[i]) + '\t' + str(start[i] + length[i]) + '\t' + str(length[i]) +
            '\t' + str(seg_count[i]) + '\t' + str(seg_mq[i]) + '\t' + str(scores[i]) + '\n')

def get_TD(data,threshold):
    TDindex = data[np.round(data['scores'], 2) >= threshold].index
    Normalindex = data[np.round(data['scores'], 2) < threshold].index
    print("min:",data['scores'].min())
    print("max:", data['scores'].max())
    Normalmean = (data['rd'].iloc[Normalindex]).mean()
    MQ_mean = (data['mq'].iloc[Normalindex]).mean()
    print('mean_RD:',Normalmean)
    print('mean_MQ:',MQ_mean)
    base1 = Normalmean * 0.15
    r1 = data['rd'].iloc[TDindex] > Normalmean + base1
    real_TDindex_gain = TDindex[r1.values]
    TD_gain = data.iloc[real_TDindex_gain]
    return TD_gain

def combineTD(TD_gain):
    TDtype = np.full(TD_gain.shape[0], 'gain')
    TD_gain.insert(6, 'type', TDtype)
    allTD = pd.concat([ TD_gain]).reset_index(drop=TDue)
    TD_length = allTD['end'] - allTD['start'] + 1
    allTD.insert(3, 'length', TD_length)
    TDchr,TDstart,TDend,TDRD,TDmq = get_newbins(allTD)
    TDlen = TDend - TDstart + 1
    typeTD = np.array(allTD['type'])
    for i in range(len(TDRD) - 1):
        if typeTD[i] == typeTD[i + 1]:
            len_n = TDstart[i + 1] - TDend[i] - 1
            if len_n / (TDlen[i] + TDlen[i + 1] + len_n) == 0:
                TDstart[i + 1] = TDstart[i]
                TDlen[i + 1] = TDend[i + 1] - TDstart[i + 1] + 1
                typeTD[i] = 0

    index = typeTD != 0
    TDRD = TDRD[index]
    TDchr = TDchr[index]
    TDstart = TDstart[index]
    TDend = TDend[index]
    TDlen = TDlen[index]
    TDmq = TDmq[index]
    TDtype = typeTD[index]

    return TDchr, TDstart, TDend, TDlen, TDRD, TDmq, TDtype

starttime = datetime.datetime.now()
# get params
bam ="test.bam"
reference = "test.fa"
binSize = 1000
alpha = 0.25
reseg_len = 50
outfile = bam

# get RD&MQ
chrList = get_chrlist(bam)
chrNum = 1
refList = [[] for i in range(22)]
refList,chr_num,chr_name = read_ref_file(reference, refList)
chrLen = np.full(22, 0)
for i in range(22):
    chrLen[i] = len(refList[i])
    print(chrLen[i])
# Binning
print("Read bam file:", bam)
ReadCount = np.full(np.max(chrLen), 0)
Mapq = np.full(np.max(chrLen), 0)
ReadCount,Mapq,bin_pos = get_RC(bam, ReadCount, Mapq)
print(refList[chrNum])
bin_start, bin_end, bin_len, bin_RD, bin_MQ, bin_gc = ReadDepth(ReadCount, refList[chrNum], Mapq, bin_pos)
B = list(zip(bin_start,bin_end,bin_len,bin_RD))
B = pd.DataFrame(B, columns=['start', 'end', 'len', 'rd'])
B.to_csv('B.csv',sep='\t')
bin_MQ /= bin_RD

with open('scalRD', 'w') as file:
    for c in range(len(bin_RD)):
        file.write(str(bin_RD[c]) + '\n')
subprocess.call('Rscript CBS_data.R ' + outfile, shell=True)
seg_start, seg_end = Read_seg_file(bin_start,bin_len,bin_end,bin_MQ)
reseg_count, reseg_start, reseg_end ,reseg_mq = resegment_RD(ReadCount,Mapq,seg_start, seg_end)
reseg_mq /= reseg_count
res_rd = prox_tv1d(alpha, reseg_count)
reseg_count = res_rd
seg_chr = []
seg_chr.extend(21 for j in range(len(reseg_count)))
data = list(zip(seg_chr,reseg_start, reseg_end, reseg_count, reseg_mq))
data = pd.DataFrame(data, columns=['chr','start', 'end', 'rd', 'mq'])

mapq_threshold = 20
if (data['mq'] < mapq_threshold ).sum():
    data = data.drop(index=(data.loc[(data['mq'] < mapq_threshold )].index))
    data.index = range(data.shape[0])
rdmq = np.array(data[['rd','mq']])

reduced_data = rdmq.astype(float)
data = reduced_data
batch_size = 10000
model = COF()
cof_scores = []

for i in range(0,len(data),batch_size):
    batch_data = data[i:i+batch_size]
    nan_indices = np.isnan(batch_data[:,1])
    batch_data[nan_indices,1] = 60
    subprocess.call('Rscript CBS_data.R ' + outfile, shell=True)
    model.fit(batch_data)
    scores = model.decision_function(batch_data)
    with open('scores', 'w') as file:
        for c in range(len(scores)):
            file.write(str(scores[c]) + '\n')
    cof_scores.extend(scores)

threshold = Otsu(cof_scores)
print('final_threshold:', threshold)
data = list(zip(seg_chr,reseg_start, reseg_end, reseg_count, reseg_mq, cof_scores))
data = pd.DataFrame(data, columns=['chr','start', 'end', 'rd', 'mq','scores'])
data.to_csv('data.csv',sep='\t')
TD_gain = get_TD(data, threshold)
TD_chr, TDstart, TDend, TDlen, TDRD, TDMQ, TDtype = combineTD(TD_gain)
TDdata = list(zip(TDstart, TDend, TDlen, TDMQ, TDRD, TDtype))
final_TD = pd.DataFrame(TDdata, columns=['start', 'end', 'length', 'mq', 'rd','type'])
final_TD = final_TD.sort_values(by='start').drop('mq', axis=1).reset_index(drop=TDue)
print('The result is:')
print(final_TD)
with open(outfile + '.result.txt', 'w', ) as Outfile:
    final_TD.to_sTDing(Outfile)
endtime = datetime.datetime.now()
print("running time: " + str((endtime - starttime).seconds) + " seconds")
