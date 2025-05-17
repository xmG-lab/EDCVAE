import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad

# 读取数据（假设数据存储在 TSV 文件）
input_file = "../data_examples/rna_expression.csv"  # 你的数据文件
df = pd.read_csv(input_file, sep="\t", index_col=0)  # 逗号分隔

# 转置，使样本作为观测值（n_obs），峰值区间作为变量（n_vars）
df = df.T

# 创建 AnnData 对象
adata = ad.AnnData(df)

# 添加样本和峰值区间信息
adata.obs_names = df.index  # 样本名
adata.var_names = df.columns  # 峰值区间

# 保存为 h5ad 文件
adata.write("rna_expression.h5ad")

# 检查转换后的数据结构
print(adata)
