# **DTDHM**

DTDHM: Detection of Tandem Duplications Based on Hybrid Methods Using Next-Generation Sequencing Data 

DOI: 10.5072/zenodo.75457

#### 1.Installation

##### 1.1 Basic requirements

- Software: Python, R, SAMtools, BWA, sambamba
- Operating System: Linux
- Python version: 3.8.5 and the higer version
- R version: 4.0.4 and the higer version
- SAMtools version: 1.11
- BWA version: 0.7.12-r1039 
- sambamba version: 0.8.2

##### 1.2 Required python packages

- numpy 1.19.2
- pysam 0.17.0
- pandas 1.1.3
- pyod 0.9.7
- matplotlib 3.3.2
- numba 0.51.2
- scikit-learn 0.23.2

##### 1.3 Required R packages

- DNAcopy

#### 2.Running software

##### 2.1 Preprocessing of input files

Usually, the following documents are required:

- A genome reference sequence fasta file. The fasta file must be indexed. You can do the following: $samtools faidx reference.fa
- A bam file from a sample. 
  The bam file needs to be de-duplicated. You can do the following: $sambamba markdup -r example.bam example_markdup.bam
- Another bam file: Extract inconsistent read pairs from the bam file. You can do the following: $samtools view -b -F 1294 example.bam > example_discordants.bam

The bam file must be sorted and indexed. You can do the following: $samtools sort example.bam; $samtools index example.bam.

##### 2.2 Operating command

**python DTDHM_v1.py [reference file] [bam file] [discordant bam] [str1]**

- reference: The path to the fasta file of the genome reference sequence used by the user.
- bam file: The path to the bam file representing the sample used by the user.
- discordant bam file: The path to the bam file containing inconsistent read pairs.
- str1:Length of read. The usual value is 100M. M means match in the CIGAR field

##### 2.3 Output file

- bam name + rusult.txt: [chromosome_name, start, end, type, length, number].
  
  Store the final results of the code.
  
- bam name + range_cigar.txt: [reference_name, pos, cigarstring, pnext, tlen, length, flag&64, flag&128, flag, query_name].
  
  Store the information required for the SR strategy. 
  
- bam name + range_discordant.txt: [reference_name, pos, cigarstring, pnext, tlen, length, flag].
  
  Store the information required for the PEM strategy. 

