# TD-COF

TD-COF, a new method for detecting tandem repeats in next generation sequencing data

1. Installation:
   
Basic requirements:

Software: Python, R, SAMtools, BWA

Operating System: Linux

Python version: 3.8.5 and the higer version

R version: 4.0.4 and the higer version

Required python packages: 

- sys
- numpy
- os
- datetime
- pandas
- pysam
- subprocess
- pyod
- numba
- imblearn

Required R packages:

- DNAcopy

2. Running software:

2.1 Preprocessing of input files:

Usually, the following documents are required:

A genome reference sequence fasta file.
A bam file from a  sample.

The bam file  must be indexed. You can do the following:
$samtools index example_sorted.bam
2.2 Operating command:

python TD-COF.py [bamfile] [reference]  

bamfile: The path to the bam file representing the sample used by the user.

reference: The path to the fasta file of the genome reference sequence used by the user.

