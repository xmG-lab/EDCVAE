import os
import argparse
import random
import torch
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from transformers import AutoTokenizer, BertModel, BertConfig

# 解析参数
parser = argparse.ArgumentParser(description="Extract open chromatin regions and generate DNABERT-2 embeddings.")
parser.add_argument("--genome", "-g", required=False, default="genomic.fna", help="参考基因组FASTA文件")
parser.add_argument("--bed", "-b", required=False, default="peak.bed", help="染色质开放区BED文件")
parser.add_argument("--out_dir", "-o", default="data_preprocess", help="输出目录")
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.out_dir, exist_ok=True)

# Step 1: 提取染色质开放区并保存为FASTA文件
print("Loading genome FASTA...")
genome = {}
with open(args.genome, "r") as f:
    for record in SeqIO.parse(f, "fasta"):
        genome[record.id] = str(record.seq).upper()

print("Extracting sequences from BED...")
sequences = []
with open(args.bed, "r") as f:
    for line in f:
        chrom, start, end = line.strip().split()[:3]
        start, end = int(start), int(end)
        if chrom in genome and end <= len(genome[chrom]):
            seq = genome[chrom][start:end]
            if len(seq) >= 30:  # DNABERT最小Kmer要求
                sequences.append(SeqRecord(Seq(seq), id=f"{chrom}_{start}_{end}", description=""))

fasta_output = os.path.join(args.out_dir, "open_regions.fa")
SeqIO.write(sequences, fasta_output, "fasta")
print(f"Saved open regions to {fasta_output}")

# Step 2: 载入DNABERT-2模型与Tokenizer并生成正负样本NPY文件
config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
model = BertModel.from_pretrained("zhihan1996/DNABERT-2-117M", config=config)
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

output_dir = os.path.join(args.out_dir, "embeddings")
os.makedirs(output_dir, exist_ok=True)

def generate_negative_sample(dna_sequence):
    dna_list = list(dna_sequence)
    random.shuffle(dna_list)
    return "".join(dna_list)

print("Processing sequences and generating embeddings...")
for record in SeqIO.parse(fasta_output, "fasta"):
    dna_sequence = str(record.seq)
    record_id = record.id.replace(":", "_").replace("-", "_")

    # 正样本编码
    inputs = tokenizer(dna_sequence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        hidden_states = model(**inputs)[0]
    pos_filename = os.path.join(output_dir, f"pos_{record_id}.npy")
    np.save(pos_filename, hidden_states[0].numpy())

    # 负样本编码
    negative_sample = generate_negative_sample(dna_sequence)
    inputs_neg = tokenizer(negative_sample, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        hidden_states_neg = model(**inputs_neg)[0]
    neg_filename = os.path.join(output_dir, f"neg_{record_id}.npy")
    np.save(neg_filename, hidden_states_neg[0].numpy())

    print(f"Saved positive sample to {pos_filename} and negative sample to {neg_filename}")

print("Preprocessing completed.")