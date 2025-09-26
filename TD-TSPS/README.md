# TD-TSPS

TD-TSPS: A hybrid strategy method for TD detection based on two-step progressive segmentation

1. Installation:

Basic requirements:

Software: Python, R, SAMtools, BWA

Operating System: Linux

Python version: 3.8.5 and the higer version

R version: 4.0.4 and the higer version

Required python packages: 

- numpy
- pandas
- pysam
- subprocess
- pyod
- numba
- imblearn
- sys
- os

Required R packages:

- DNAcopy

2. Running software:

2.1 Preprocessing of input files:

Usually, the following documents are required:

A genome reference sequence fasta file.

A bam file from a  sample.

The bam file  must be indexed. 

You can do the following: $samtools index sample.sorted.bam

2.2 Operating command:

python TD-TSPS.py [reference] [bam file] [discordant bam file] [str1] 

reference: The path to the fasta file of the genome reference sequence used by the user.

bam file: The path to the bam file representing the sample used by the user.

discordant bam file: The path to the bam file containing inconsistent read pairs.

str1: Length of read. The usual value is 100M. M means match in the CIGAR field.

2.3 Output file

bam name + result.txt: [chromosome_name, start, end, length, number].

Store the final results of the code.

bam name + range_cigar.txt

Store the information required for the SR strategy.

bam name + range_discordant.txt

Store the information required for the PEM strategy.

