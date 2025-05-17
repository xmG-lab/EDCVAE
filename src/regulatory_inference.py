import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scglue
import seaborn as sns
import subprocess
from IPython import display
from matplotlib import rcParams
from networkx.algorithms.bipartite import biadjacency_matrix
from networkx.drawing.nx_agraph import graphviz_layout

scglue.plot.set_publication_params()
rcParams['figure.figsize'] = (4, 4)

rna = ad.read_h5ad("rna-emb.h5ad")
atac = ad.read_h5ad("atac-emb.h5ad")
guidance_hvf = nx.read_graphml("guidance-hvf.graphml.gz")

rna.var["name"] = rna.var_names
atac.var["name"] = atac.var_names

genes = rna.var.query("highly_variable").index
peaks = atac.var.query("highly_variable").index

features = pd.Index(np.concatenate([rna.var_names, atac.var_names]))


feature_embeddings = np.concatenate([rna.varm["X_glue"], atac.varm["X_glue"]])


skeleton = guidance_hvf.edge_subgraph(
    e for e, attr in dict(guidance_hvf.edges).items()
    if attr["type"] == "fwd"
).copy()


reginf = scglue.genomics.regulatory_inference(
    features, feature_embeddings,
    skeleton=skeleton, random_state=0
)

gene2peak = reginf.edge_subgraph(
    e for e, attr in dict(reginf.edges).items()
    if attr["qval"] < 0.05
)

scglue.genomics.Bed(atac.var).write_bed("peaks.bed", ncols=3)
scglue.genomics.write_links(
    gene2peak,
    scglue.genomics.Bed(rna.var).strand_specific_start_site(),
    scglue.genomics.Bed(atac.var),
    "gene2peak.links", keep_attrs=["score"]
)

df = pd.read_csv("gene2peak.links", sep="\t", header=None)
df.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "score"]

# 计算中点
df["mid1"] = (df["start1"] + df["end1"]) / 2
df["mid2"] = (df["start2"] + df["end2"]) / 2

# 过滤掉中点相等的行
df_filtered = df[df["mid1"] != df["mid2"]]

# 保存新文件
df_filtered.drop(columns=["mid1", "mid2"]).to_csv("gene2peak.filtered.links", sep="\t", header=False, index=False)


# 读取已过滤的links文件
df = pd.read_csv("gene2peak.filtered.links", sep="\t", header=None)
df.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "score"]

# 构建一个 link 位置对的标识（也可以考虑只看 atac 端）
df["link_pos"] = df["chrom2"] + ":" + df["start2"].astype(str) + "-" + df["end2"].astype(str)


loc = rna.var.loc["AT1G01040"]
chrom = loc["chrom"]
chromLen = loc["chromEnd"] - loc["chromStart"]
chromStart = max(0, loc["chromStart"] - chromLen)
chromEnd = loc["chromEnd"] + chromLen

cmd = f"pyGenomeTracks --tracks tracks.ini --region {chrom}:{chromStart}-{chromEnd} --outFileName tracks.png"

subprocess.run(cmd, shell=True, check=True)

display.Image("tracks.png")