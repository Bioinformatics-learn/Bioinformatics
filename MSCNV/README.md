# MSCNV

MSCNV: A New Method for Detecting Copy Number Variations from Next-Generation Sequencing Data Based on Multistrategy Integration 

#### 1.Installation

##### 1.1 Basic requirements

- Software: Python, SAMtools, BWA, sambamba
- Operating System: Linux
- Python version: 3.8.5 and the higer version
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

#### 2.Running software

##### 2.1 Preprocessing of input files

Usually, the following documents are required:

- A genome reference sequence fasta file. The fasta file must be indexed. You can do the following: $samtools faidx reference.fa
- A bam file from a sample. 
  The bam file needs to be de-duplicated. You can do the following: $sambamba markdup -r example.bam example_markdup.bam
- A discordant bam file: Extract inconsistent read pairs from the bam file. You can do the following: $samtools view -b -F 1294 example.bam > example_discordants.bam
- A split bam file: Extract split read from the bam file. You can do the following: $samtools view -h example.bam | extractSplitReads_BwaMem -i stdin | samtools view -Sb -  > example_split.bam

The bam file must be sorted and indexed. You can do the following: $samtools sort example.bam; $samtools index example.bam.

##### 2.2 Operating command

**python MSCNV_v1.py [reference file] [bam file] [split bam] [discordant bam]**

- reference file: The path to the fasta file of the genome reference sequence used by the user.
- bam file: The path to the bam file representing the sample used by the user.
- split bam file: The path to the bam file containing split read.
- discordant bam file: The path to the bam file containing inconsistent read pairs.

##### 2.3 Output file

- bam name + rusult.txt: [chromosome_name, start, end, type].
  
  Store the final results of the code.
  
- bigrangeciagrresult: [reference_name, pos, cigarstring, isize].
  
  Store the information required for the SR strategy. 
  
- disrangesizeresult: [reference_name, pos, cigarstring, isize, length].
  
  Store the information required for the PEM strategy. 

