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
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.out_dir, exist_ok=True)
embedding_dir=os.path.join(args.out_dir, "embeddings")
os.makedirs(embedding_dir, exist_ok=True)

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
           if len(seq) >= 30:  # DNABERT最小K-mer要求
               sequences.append(SeqRecord(Seq(seq), id=f"{chrom}_{start}_{end}", description=""))
        if len(seq)!=WINDOW_SIZE:
            continue
print("sequences:",len(seq))


fasta_output = os.path.join(args.out_dir, "open_regions.fa")
SeqIO.write(sequences, fasta_output, "fasta")
print(f"Saved open regions to {fasta_output}")

# Step 2: Repeat filtering
if args.repeat is not None:
    print( "Loading repeat regions..." )
    repeat_regions={}
    with open(args.repeat,"r") as f:
        for line in f:
            chrom,start,end=line.strip().split()[:3]
            start=int(start)
            end=int(end)
            if chrom not in repeat_regions:
                repeat_regions[chrom]=[]
            repeat_regions[chrom].append( (start, end) )

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
    for item in positive_sequences:
        if repeat_fraction(
            item["chrom"],
            item["start"],
            item["end"]
        ) <= MAX_REPEAT_FRACTION:
            filtered.append(item)
    sequences=filtered
print("After repeat filtering:", len(sequences))

# Step 3: 去除完全重复和reverse complement重复
print("Removing duplicate sequences...")
def canonical(seq):
    rc=str(Seq(seq).reverse_complement())
    return min(seq, rc)
seen=set()
unique=[]
for item in sequences:
    key=canonical(item["sequence"])
    if key in seen:
        continue
    seen.add(key)
    unique.append(item)
sequences=unique
print("After duplicate removal:", len(sequences)
)
# 重新保存最终fasta output
SeqIO.write([SeqRecord(Seq(x["sequence"]), id=x["id"],description="") for x in positive_sequences], fasta_output, "fasta")

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
SeqIO.write(negative_sequences, negative_fasta, "fasta")
print("Saved negative regions:", negative_fasta)


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
