from itertools import chain

import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import networkx as nx
import random
import scglue

# ------------------- 读取数据 -------------------
rna_adata = ad.read_h5ad("../data_examples/rna_expression.h5ad")
atac_adata = ad.read_h5ad("../data_examples/atac_expression.h5ad")

print("RNA数据概览：", rna_adata)
print("ATAC数据概览：", atac_adata)

# ------------------- 预处理RNA数据 -------------------
rna_adata.layers["counts"] = rna_adata.X.copy()
sc.pp.highly_variable_genes(rna_adata, n_top_genes=2000, flavor="seurat_v3", layer="counts")
sc.pp.normalize_total(rna_adata)
sc.pp.log1p(rna_adata)
sc.pp.scale(rna_adata)
sc.tl.pca(rna_adata, n_comps=20, svd_solver="auto")

# 基因注释：利用 scglue 提供的接口，将结果写入 rna_adata.var
scglue.data.get_gene_annotation(
    rna_adata, 
    gtf="genomic.gtf",
    # gtf_by="gene_name"
    gtf_by="gene_id"
)

# 让基因名唯一，并过滤掉没有位置信息的基因
rna_adata.var_names_make_unique()
valid_genes = rna_adata.var.dropna(subset=["chrom", "chromStart", "chromEnd"]).index
valid_genes = np.intersect1d(rna_adata.var_names, valid_genes)
rna_adata = rna_adata[:, valid_genes].copy()

# 确保坐标为整数类型
rna_adata.var["chromStart"] = rna_adata.var["chromStart"].astype(int)
rna_adata.var["chromEnd"] = rna_adata.var["chromEnd"].astype(int)

# ------------------- 预处理ATAC数据 -------------------
if not atac_adata.var_names.str.contains(":").any():
    print("检测到ATAC的var_names不是峰ID格式，正在修复...")
    atac_adata.var_names = atac_adata.var.apply(
        lambda x: f"{x['chrom']}:{x['start']}-{x['end']}", axis=1
    )
split = atac_adata.var_names.str.split(r"[:-]")
atac_adata.var["chrom"] = split.map(lambda x: x[0])
atac_adata.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
atac_adata.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
atac_df = atac_adata.var.copy()

# ------------------- 构建RNA锚定的调控图 -------------------

G = scglue.genomics.rna_anchored_guidance_graph(rna_adata, atac_adata)

print("构建的图包含节点数：", G.number_of_nodes())
print("构建的图包含边数：", G.number_of_edges())

# ------------------- 保存处理后的数据 -------------------
# 保存RNA数据到 h5ad 文件（使用gzip压缩）

# 检查 obs 中是否存在 'note' 列，如果存在，则将其转换为字符串类型
if 'note' in rna_adata.obs.columns:
    rna_adata.obs['note'] = rna_adata.obs['note'].astype(str)

# 同样检查 var 中是否存在 'note' 列
if 'note' in rna_adata.var.columns:
    rna_adata.var['note'] = rna_adata.var['note'].astype(str)

# 或者将 var 中所有元素转换为字符串（如果确定不会影响后续分析）
# rna_adata.var = rna_adata.var.applymap(lambda x: str(x) if not pd.isnull(x) else x)

print(rna_adata.var.dtypes)

if 'exception' in rna_adata.var.columns:
    rna_adata.var.drop(columns='exception', inplace=True)

if 'part' in rna_adata.var.columns:
    rna_adata.var.drop(columns='part', inplace=True)

# 保存 RNA 数据到 h5ad 文件
rna_adata.write("rna-pp.h5ad", compression="gzip")

print("RNA数据已保存为 rna-pp.h5ad")

# 保存ATAC数据到 h5ad 文件（使用gzip压缩）
atac_adata.write("atac-pp.h5ad", compression="gzip")
print("ATAC数据已保存为 atac-pp.h5ad")

# 保存构建的指导图到 graphml 格式文件（gzip压缩）
nx.write_graphml(G, "guidance.graphml.gz")
print("指导图已保存为 guidance.graphml.gz")


# 保存构建的高变异指导图到 guidance-hvf.graphml 格式文件（gzip压缩）
guidance_hvf = G.subgraph(chain(
    rna_adata.var.query("highly_variable").index,
    atac_adata.var.query("highly_variable").index
)).copy()

print("构建的图包含节点数：", guidance_hvf.number_of_nodes())
print("构建的图包含边数：", guidance_hvf.number_of_edges())

nx.write_graphml(guidance_hvf, "guidance-hvf.graphml.gz")
print("高变异指导图已保存为 guidance-hvf.graphml.gz")