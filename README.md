# EDCVAE
EDCVAE（Epigenomic Deep Chromatin Variational AutoEncoder）是一个基于深度学习的模型，旨在整合单细胞 RNA-seq 和 ATAC-seq 数据，结合 DNA-BERT2 预训练模型，对拟南芥（Arabidopsis thaliana）的顺式调控区域进行预测和分析。该项目融合了多模态数据处理、深度学习建模和生物信息学分析，提供了从数据预处理到结果可视化的完整流程。
EDCVAE/
├── data_examples/           # 示例数据
│   ├── atac-pp.h5ad
│   ├── atac_expression.csv
│   ├── atac_expression.h5ad
│   ├── rna-pp.h5ad
│   ├── rna_expression.csv
│   └── rna_expression.h5ad
├── result/                  # 模型预测结果
│   ├── gene2peak.links
│   ├── motif/
│   │   └── filter_meme.txt
│   └── predict_OCR/
│       ├── OCR_A.tha.txt
│       ├── OCR_O.sat.txt
│       ├── OCR_P.tri.txt
│       └── OCR_Z.may.txt
├── src/                     # 源代码
│   ├── VAE_emb.py
│   ├── data_atac.py
│   ├── data_preprocess.py
│   ├── data_rna.py
│   ├── guidance_hvf.py
│   ├── model_OCR.py
│   ├── motif.py
│   └── regulatory_inference.py
├── genomic.fna              # 拟南芥参考基因组序列
├── genomic.gtf              # 拟南芥基因注释文件
├── peak.bed                 # 染色质开放区域（ATAC-seq peaks）
├── tracks.ini               # 顺式调控区域推断配置文件
├── .gitattributes           # Git LFS 配置文件
└── README.md                # 项目说明文档
# 项目依赖环境
本项目基于 Ubuntu Linux 操作系统，使用 Conda 管理虚拟环境，核心依赖如下：

Python 3.10.14

PyTorch 2.3.0（CUDA 11.8）

CUDA Toolkit 11.7

cuDNN 8.0.5

Transformers 4.28.1

scikit-learn 1.6.1

h5py 3.12.1

Biopython 1.78

scglue 0.3.1

确保安装上述依赖后，使用 torch.cuda.is_available() 验证 GPU 是否可用。
# 数据处理
在数据预处理阶段，项目使用了 DNA-BERT2-117M 预训练模型，对 DNA 序列进行嵌入表示。DNA-BERT2 是一个基于 Transformer 的基因组基础模型，采用了 Byte Pair Encoding (BPE) 代替传统的 k-mer 分词方法，支持无限输入长度，并在多物种基因组理解任务中表现出色。而csv文件格式的rna和atc数据则需要用src文件当中的data_rna.py和data_atac.py处理。
# 数据说明
输入数据
data_examples/：包含预处理后的单细胞 RNA-seq 和 ATAC-seq 数据，格式为 .h5ad 和 .csv。

genomic.fna：参考基因组序列。

genomic.gtf：基因注释文件。

peak.bed：通过 ATAC-seq 实验获得的染色质开放区域（peaks）。

# 输出结果
result/gene2peak.links：模型预测的基因与峰值之间的关联信息。

result/motif/filter_meme.txt：模体分析结果文件。

result/predict_OCR/：模型预测的染色质开放区域文件，分别对应不同的样本或条件。
# 使用说明
## 克隆项目仓库：
git clone https://github.com/xmG-lab/EDCVAE.git
cd EDCVAE
## 创建并激活 Conda 虚拟环境，安装依赖：
conda create -n edcvae_env python=3.10.14
conda activate edcvae_env
pip install -r requirements.txt
## 运行数据预处理脚本：
python src/data_preprocess.py
## 训练模型并进行预测：
python src/model_OCR.py
## 进行模体分析和顺式调控区域推断：
python src/motif.py
python src/regulatory_inference.py
# 联系方式
如有任何问题或建议，欢迎通过 Issues 提交，或联系项目维护者：

邮箱：2042766474@qq.com

GitHub：https://github.com/xmG-lab/EDCVAE
