# Required Modules
from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser

# Specify GPU(s)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = ArgumentParser(description="Motif Visualization")
parser.add_argument("--nb_filter1", "-n1", default=200, type=int, required=False, help="Number of filters in first layer of convolution.")
parser.add_argument("--filter_len1", "-fl1", default=19, type=int, required=False, help="Length of filters in first layer of convolution.")
parser.add_argument("--out", "-o", default="Ory", required=False, help="Prefix of the output file")
parser.add_argument("--nb_filter2", "-n2", default=100, type=int, required=False, help="Number of filters in second layer of convolution.")
parser.add_argument("--filter_len2", "-fl2", default=11, type=int, required=False, help="Length of filters in second layer of convolution.")
parser.add_argument("--dropout", "-d", default=0.6, type=float, required=False, help="Dropout rate. (default is 0.6)")
parser.add_argument("--hidden", "-hd", default=200, type=int, required=False, help="Units in the fully connected layer. (default is 200)")
args = parser.parse_args()

# Load data and model paths
os.chdir("../data_preprocess")
input_file_path = os.getcwd()
try:
    data_train = np.load(os.path.join(input_file_path, "data_train.npy"))
except FileNotFoundError:
    raise FileNotFoundError("训练数据文件 'data_train.npy' 未找到，请确保文件存在于 '../data_preprocess' 目录中。")
file = [f for f in os.listdir() if f.endswith('train.txt')][0]
motif_input_file = os.path.join(input_file_path, file)

os.chdir("../model")
input_model_path = os.getcwd()
model_file = [f for f in os.listdir() if f.endswith('.pth')][0]
model_path = os.path.join(input_model_path, model_file)

# Define the model
class CharPlantCNN(nn.Module):
    def __init__(self):
        super(CharPlantCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=args.nb_filter1, kernel_size=args.filter_len1, padding=args.filter_len1 // 2)
        self.conv2 = nn.Conv1d(in_channels=args.nb_filter1, out_channels=args.nb_filter2, kernel_size=args.filter_len2, padding=args.filter_len2 // 2)
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(in_features=args.nb_filter2 * 512, out_features=args.hidden)
        self.fc2 = nn.Linear(in_features=args.hidden, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  # 从 (batch, seq_len, embedding_dim) 转换为 (batch, embedding_dim, seq_len)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Prepare motif data
os.chdir("../motif")
output_dir = f"{args.out}_motif"
os.makedirs(output_dir, exist_ok=True)

sequence = [line.strip().split('\t')[1] for line in open(motif_input_file, 'r')]

# Define motif processing functions
def print_pwm(f, filter_idx, filter_pwm, nsites):
    if nsites < 10:
        return
    print(f'MOTIF filter{filter_idx}', file=f)
    print(f'letter-probability matrix: alength= 4 w= {filter_pwm.shape[0]} nsites= {nsites}', file=f)
    for i in range(filter_pwm.shape[0]):
        print(f'{filter_pwm[i][0]:.4f} {filter_pwm[i][1]:.4f} {filter_pwm[i][2]:.4f} {filter_pwm[i][3]:.4f}', file=f)
    print('', file=f)

def logo_kmers(filter_outs, filter_size, seqs, filename, maxpct_t=0.7):
    all_outs = np.ravel(filter_outs)
    raw_t = maxpct_t * all_outs.max()
    half_filter = filter_size // 2
    with open(filename, 'w') as f:
        for i in range(filter_outs.shape[0]):
            for j in range(filter_outs.shape[1]):
                if filter_outs[i, j] > raw_t:
                    start = max(0, j - half_filter)
                    end = min(len(seqs[i]), j + half_filter + 1)
                    kmer = seqs[i][start:end]
                    if len(kmer) < filter_size:
                        continue
                    print(f'>{i}_{j}', file=f)
                    print(kmer, file=f)

def make_filter_pwm(filter_fasta):
    nts = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    pwm_counts = []
    nsites = 4  # 初始值为4，表示背景频率
    for line in open(filter_fasta):
        if line.startswith('>'):
            continue
        seq = line.rstrip()
        nsites += 1
        if not pwm_counts:
            pwm_counts = [np.array([1.0] * 4) for _ in range(len(seq))]
        for i, nt in enumerate(seq):
            try:
                pwm_counts[i][nts[nt]] += 1
            except KeyError:
                pwm_counts[i] += np.array([0.25] * 4)  # 对于非ACGT的字符，均匀分配
    pwm_freqs = np.array([[pwm[i] / nsites for i in range(4)] for pwm in pwm_counts])
    return pwm_freqs, nsites - 4  # 减去初始的4

def meme_header(f, seqs):
    nts = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    nt_counts = [1] * 4
    for seq in seqs:
        for nt in seq:
            try:
                nt_counts[nts[nt]] += 1
            except KeyError:
                pass
    nt_sum = float(sum(nt_counts))
    nt_freqs = [nt_counts[i] / nt_sum for i in range(4)]
    print('MEME version 4\n', file=f)
    print('ALPHABET= ACGT\n', file=f)
    print(f'Background letter frequencies:\nA {nt_freqs[0]:.4f} C {nt_freqs[1]:.4f} G {nt_freqs[2]:.4f} T {nt_freqs[3]:.4f}\n', file=f)

# Load the PyTorch model
model = CharPlantCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define a function to get the output of a specific layer
def get_layer_output(model, x, layer_index):
    with torch.no_grad():
        x = x.to(device)
        modules = list(model.children())
        for i in range(layer_index + 1):
            x = modules[i](x)
        return x

# Forward pass and motif extraction
print("Original data shape:", data_train.shape)
data = torch.tensor(data_train, dtype=torch.float32)
print("Data shape after tensor conversion:", data.shape)

final_output = []

with torch.no_grad():
    batch_size = 3000
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size].to(device)
        # 获取第一层卷积层后的ReLU激活输出
        cnn_output = get_layer_output(model, batch_data, 1)  # layer=1 对应 self.relu(self.conv1(x))
        cnn_output = cnn_output.transpose(1, 2).cpu().numpy()  # 调整为 (batch, seq_len, out_channels)
        final_output.append(cnn_output)

final_output = np.concatenate(final_output, axis=0)
print("Final output shape:", final_output.shape)

# Save motifs to files
meme_file = os.path.join(output_dir, "filter_meme.txt")
with open(meme_file, 'w') as meme:
    meme_header(meme, sequence)
    for i in range(args.nb_filter1):
        filter_outs = final_output[:, :, i]
        fasta_file = os.path.join(output_dir, f"filter_{i}.fa")
        logo_kmers(filter_outs, args.filter_len1, sequence, fasta_file)
        filter_pwm, nsites = make_filter_pwm(fasta_file)
        print_pwm(meme, i, filter_pwm, nsites)