# MicroDNA Map

# Installation
We recommend installing via Conda with Python version 3.9.
``` 
# Create a Python environment
conda create -n microdna python==3.9
conda activate microdna

# Install dependencies
pip install -r requirements.txt

# Install PyTorch manually (version ≥ 1.8, CUDA ≥ 11.7 recommended)
# Visit https://pytorch.org/get-started/previous-versions/ to select a suitable version
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install CNVkit (check https://github.com/etal/cnvkit for guidance)
# Tip: Testing shows best compatibility with Python 3.9
pip install cnvkit
``` 

# Quick Test
``` 
conda activate YOUR_ENV_NAME
cd YOUR_DIR_PATH
python run.py --pattern long_segment
``` 
This will automatically process .fa files in ./identify/long_segment_to_identify/to identify microDNAs and output results to the same directory. If the FASTA headers contain correct chromosomal positions, the results will include .bed localization files and .fasta sequence files.
``` 
# View more options for run.py
python run.py -h
``` 

# Starting from FASTQ Files

1. Place hg19.fain the ./cnvkit_do directory.

2. Place paired *.fastq1 and *.fastq2 files in ./cnvkit_do (these can be obtained from SRA files using fasterq-dump --split-3 *.sra).

3. Run MicroDNA_Map_batch.py in the Conda environment with dependencies installed:
``` 
python MicroDNA_Map_batch.py
``` 

#### Pipeline Overview:
Steps in cnvkit_tool.py:

0. Glob all FASTQ files in ./cnvkit_do

1. Check for the existence of the hg19.faindex; build it with Bowtie2 if missing

2. Validate FASTQ file integrity

3. Align FASTQ reads to the reference genome using Bowtie2

4. Call CNVs using CNVkit

Steps in cnvkit_run.py:

0. Verify the completeness of previous step outputs

1. Load CNVkit results

2. Extract CNV segments using Samtools

3. Execute run.pyfor microDNA identification

Finally, results are consolidated and intermediate files are cleaned up.

# Training
### Data pre-processing
Download *_RAWfiles from the NCBI GEO database to extract microDNA positional information.

Use ./preprocessing/count*.py to extract eccDNA sequences and ./preprocessing/count_other*.py for otherDNA sequences. These scripts call SamtoolBash.sh to perform sequence segmentation.

Note: Datasets may vary in format. Manually adjust preprocessing script parameters as needed.

### model training
1. Place data in the ./datasetsdirectory

2. Adjust training parameters in train.py(e.g., epochs, learning rate scheduler)

3. Run the training script:
``` 
python train.py
``` 
### Independent Testing
Place test data in ./datasets, then run the following to obtain evaluation metrics:
``` 
python verification.py
``` 
Generate ROC curves with:
``` 
python ROC_draw.py
``` 
<br />
<br />


#### Additional Notes
./Additional_toolscontains bioinformatics utilities, but the code was not designed for general use and is provided for reference only.

./other_modelincludes experimental model implementations; review the code carefully before use.
