# EDCVAE
EDCVAE (Epigenomic Deep Chromatin Variational AutoEncoder) is a deep learning-based model designed to integrate RNA-seq and ATAC-seq data, leveraging the DNA-BERT2 pretrained model to predict and analyze cis-regulatory regions in Arabidopsis thaliana. The project combines multimodal data processing, deep learning modeling, and bioinformatics analysis, providing a comprehensive workflow from data preprocessing to result visualization.
# Dependencies
This project is based on the Ubuntu Linux operating system and uses Conda for virtual environment management. The core dependencies are as follows:  
  __Python 3.10.14__  
  __PyTorch 2.3.0（CUDA 11.8）__  
  __CUDA Toolkit 11.7__  
  __cuDNN 8.0.5__  
  __Transformers 4.28.1__  
  __scikit-learn 1.6.1__  
  __h5py 3.12.1__  
  __Biopython 1.78__  
  __scglue 0.3.1__  
After installing the dependencies, verify GPU availability using torch.cuda.is_available().  
# Data Processing
During the data preprocessing stage, the project utilizes the DNA-BERT2-117M pretrained model to generate embeddings for DNA sequences. DNA-BERT2 is a Transformer-based genomic foundation model that employs Byte Pair Encoding (BPE) instead of traditional k-mer tokenization, supports unlimited input lengths, and excels in multi-species genomic understanding tasks. The RNA and ATAC data in CSV format should be processed using data_rna.py and data_atac.py in the src directory.
# Data Description
## Input Data
__data_examples/:__ Contains preprocessed single-cell RNA-seq and ATAC-seq data in .h5ad and .csv formats.  
__genomic.fna:__ Reference genome sequence.  
__genomic.gtf:__ Gene annotation file.  
__peak.bed:__ Chromatin accessible regions (peaks) obtained from ATAC-seq experiments.  
## Output Results
__result/gene2peak.links:__ Predicted associations between genes and peaks.   
__result/motif/filter_meme.txt:__ Motif analysis result file.    
__result/predict_OCR/:__ Predicted chromatin accessible region files, corresponding to different samples or conditions.    
# Usage Instructions
## Clone the repository:

```
git clone https://github.com/xmG-lab/EDCVAE.git    
cd EDCVAE
```
## Create and activate the Conda virtual environment, and install dependencies:
```
conda create -n edcvae_env python=3.10.14    
conda activate edcvae_env    
pip install -r requirements.txt
```
## Run the data preprocessing script:
```
python src/data_preprocess.py
```
## Train the model and perform predictions:
```
python src/model_OCR.py
```
## Perform motif analysis and cis-regulatory region inference:
```
python src/motif.py    
python src/regulatory_inference.py
```
# Contact
For any questions or suggestions, please submit an issue on GitHub or contact the project maintainer:  
Email:2042766474@qq.com  
GitHub:https://github.com/xmG-lab/EDCVAE
