import os
import argparse
import random
import torch
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from transformers import AutoTokenizer, BertModel, BertConfig
from embedding import load_dnabert2, generate_base_embedding, save_embedding


# 解析参数
parser = argparse.ArgumentParser(description="Extract open chromatin regions and generate DNABERT-2 embeddings.")
parser.add_argument("--genome", "-g", required=False, default="genomic.fna", help="参考基因组FASTA文件")
parser.add_argument("--bed", "-b", required=False, default="peak.bed", help="染色质开放区BED文件")
parser.add_argument("--out_dir", "-o", default="data_preprocess", help="输出目录")
parser.add_argument("--repeat", "-r", default=None, help="重复序列区域BED文件；不提供则跳过repeat filtering")
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.out_dir, exist_ok=True)
embedding_dir=os.path.join(args.out_dir, "embeddings")
os.makedirs(embedding_dir, exist_ok=True)
existing_embeddings = [
    f for f in os.listdir(embedding_dir)
    if f.endswith(".npy")
    and (
        f.startswith("pos_")
        or f.startswith("neg_")
    )
]

if existing_embeddings:
    raise RuntimeError(
        f"{embedding_dir} already contains "
        f"{len(existing_embeddings)} embedding files. "
        "Please use a new output directory or remove "
        "the old embeddings before preprocessing."
    )

# 参数设置
WINDOW_SIZE = 500
MAX_REPEAT_FRACTION = 0.2
DNA_ALPHABET = set("ACGT")

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
        if line.startswith("#"):
            continue
        if not line.strip():
            continue
            
        chrom, start, end = line.strip().split()[:3]
        start, end = int(start), int(end)
        if chrom not in genome:
            continue
        center=(start+end)//2
        window_start=center-WINDOW_SIZE//2
        window_end=center+WINDOW_SIZE//2
        if window_start <0:
            continue
        if window_end > len(genome[chrom]):
            continue
        seq = genome[chrom][window_start:window_end]
           # 必须严格保证输入窗口为500 bp
        if len(seq) != WINDOW_SIZE:
            continue
        if set(seq) - DNA_ALPHABET:
            continue
        # 保存peak及实际500-bp窗口坐标
        record.annotations["chrom"] = chrom
        record.annotations["peak_start"] = start
        record.annotations["peak_end"] = end
        record.annotations["window_start"] = window_start
        record.annotations["window_end"] = window_end
        sequences.append(record)
print("sequences:", len(sequences))
fasta_output = os.path.join(args.out_dir, "open_regions.fa")
SeqIO.write(sequences, fasta_output, "fasta")
print(f"Saved open regions to {fasta_output}")

# Step 2: Repeat filtering
if args.repeat is not None:
    print( "Loading repeat regions..." )
    repeat_regions={}
    with open(args.repeat, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if not line.strip():
                continue
            chrom, start, end = line.strip().split()[:3]
            start = int(start)
            end = int(end)
            if chrom not in repeat_regions:
                repeat_regions[chrom] = []
            repeat_regions[chrom].append((start, end))
    for chrom in repeat_regions:
        repeat_regions[chrom].sort(key=lambda x: x[0])
    def repeat_fraction(chrom, start, end):
        if chrom not in repeat_regions:
            return 0
        overlap=0
        for r1,r2 in repeat_regions[chrom]:
            if r2<=start:
                continue
            if r1>=end:
                break
            overlap += max(0, min(end,r2)-max(start,r1))
        return overlap/(end-start)
    filtered=[]
    for item in sequences:
        if repeat_fraction(item.annotations["chrom"], item.annotations["window_start"], item.annotations["window_end"]) <= MAX_REPEAT_FRACTION:
            filtered.append(item)
    sequences = filtered
print("After repeat filtering:", len(sequences))
# Step 3: 去除完全重复和reverse complement重复
print("Removing duplicate sequences...")
def canonical(seq):
    rc=str(Seq(seq).reverse_complement())
    return min(seq, rc)
seen=set()
unique=[]
for item in sequences:
    key=canonical(str(item.seq))
    if key in seen:
        continue
    seen.add(key)
    unique.append(item)
sequences=unique
print("After duplicate removal:", len(sequences))

# 重新保存最终fasta output
SeqIO.write(sequences, fasta_output, "fasta")

# Step 4: 生成dinucleotide shuffled negative
print("Generating dinucleotide shuffled negatives...")
def dinucleotide_shuffle(sequence):
    graph={
        "A":[],
        "C":[],
        "G":[],
        "T":[]
    }
    for a,b in zip(sequence[:-1],sequence[1:]):
        graph[a].append(b)
    for key in graph:
        random.shuffle(graph[key])
    stack=[sequence[0]]
    path=[]
    while stack:
        node=stack[-1]
        if graph[node]:
            stack.append(graph[node].pop())
        else:
            path.append(stack.pop())
    return "".join(reversed(path))
negative_sequences=[]
for record in sequences:
    negative_sequences.append(SeqRecord(Seq(dinucleotide_shuffle(str(record.seq))), id=record.id, description=""))
negative_fasta=os.path.join(args.out_dir, "negative_regions.fa")

# Step 5: 检查正负样本序列冲突
print("Checking positive-negative sequence conflicts...")
def check_positive_negative_conflicts(sequences, negative_sequences, max_attempts=10):
    # 保存所有正样本及其reverse complement的canonical形式
    positive_set = set()
    for record in sequences:
        positive_set.add(canonical(str(record.seq)))
    filtered_sequences = []
    filtered_negative_sequences = []
    conflict_count = 0
    regenerated_count = 0
    removed_count = 0
    
    for pos_record, neg_record in zip(sequences, negative_sequences):
        negative_sequence = str(neg_record.seq)
        attempt = 0
        # 如果负样本与任意正样本相同或互为reverse complement，则重新进行dinucleotide shuffle
        while (
            canonical(negative_sequence) in positive_set and attempt < max_attempts):
            negative_sequence = dinucleotide_shuffle(str(pos_record.seq))
            attempt += 1
        # 记录发生过冲突并成功重新生成的样本
        if attempt > 0:
            conflict_count += 1
        # 多次重新shuffle后仍然存在冲突，则删除整个正负样本对
        if canonical(negative_sequence) in positive_set:
            removed_count += 1
            continue
        if attempt > 0:
            regenerated_count += 1
        filtered_sequences.append(pos_record)
        filtered_negative_sequences.append(SeqRecord(Seq(negative_sequence), id=neg_record.id, description=""))

    print("Detected conflicts:", conflict_count)
    print("Successfully regenerated negatives:", regenerated_count)
    print("Removed positive-negative pairs:", removed_count)
    return (filtered_sequences, filtered_negative_sequences)
sequences, negative_sequences = check_positive_negative_conflicts(sequences,negative_sequences)

print("Final positive samples:", len(sequences))
print("Final negative samples:", len(negative_sequences))
SeqIO.write(negative_sequences, negative_fasta, "fasta")
print("Saved final negative regions:", negative_fasta)

# Step 6:创建dataset
dataset=[]
for pos,neg in zip(sequences, negative_sequences):
    dataset.append({"id":"pos_"+pos.id, "sequence":str(pos.seq)})
    dataset.append({"id":"neg_"+neg.id, "sequence":str(neg.seq)})

# Step 7:载入DNABERT-2

print("Loading DNABERT-2...")
tokenizer, model, device = load_dnabert2()

# Step 8:生成正负样本npy

print("Generating embeddings...")
for row in dataset:
    embedding = generate_base_embedding(row["sequence"], tokenizer, model, device)
    if embedding.shape != (500,768):
        raise RuntimeError(f"Unexpected embedding shape: {embedding.shape}")
    save_embedding(os.path.join(embedding_dir, row["id"].replace(":", "_")+".npy"), embedding)
print("Preprocessing completed.")
