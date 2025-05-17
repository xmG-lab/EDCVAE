import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
import pyfiglet
import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description="Training Charplant CNN model")
parser.add_argument("--epochs", "-e", default=150, type=int, required=False,
                    help="Number of epochs. (default is 150)")
parser.add_argument("--patience", "-p", default=150, type=int, required=False,
                    help='Number of epochs for early stopping. (default is 20)')
parser.add_argument("--learningrate", "-lr", default=0.001, type=float, required=False,
                   help='Learning rate. (default is 0.001)')
parser.add_argument("--batch_size", "-b", default=28, type=int, required=False,
                    help="Batch Size. (default is 128)")
parser.add_argument("--dropout", "-d", default=0.6, type=float, required=False,
                    help="Dropout rate. (default is 0.6)")
parser.add_argument("--nb_filter1", "-n1", default=200, type=int, required=False,
                    help="Number of filters in first layer of convolution. (default is 200)")
parser.add_argument("--nb_filter2", "-n2", default=100, type=int, required=False,
                    help="Number of filters in second layer of convolution. (default is 100)")
parser.add_argument("--filter_len1", "-fl1", default=19, type=int, required=False,
                    help="Length of filters in first layer of convolution. (default is 19)")
parser.add_argument("--filter_len2", "-fl2", default=11, type=int, required=False,
                    help="Length of filters in second layer of convolution. (default is 11)")
parser.add_argument("--hidden", "-hd", default=200, type=int, required=False,
                    help="Units in the fully connected layer. (default is 200)")
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from 'data' folder
print("loading data from 'data' folder...")
data_dir = '/data/gxmst/Finally/DNAbert2+CNN/data_preprocess/embeddings'
pos_files = [f for f in os.listdir(data_dir) if f.startswith('pos_')]
neg_files = [f for f in os.listdir(data_dir) if f.startswith('neg_')]

# Load embeddings and labels
data = []
labels = []

for pos_file in pos_files:
    pos_data = np.load(os.path.join(data_dir, pos_file))
    data.append(pos_data)
    labels.append(1)  # Positive sample label

for neg_file in neg_files:
    neg_data = np.load(os.path.join(data_dir, neg_file))
    data.append(neg_data)
    labels.append(0)  # Negative sample label

# data = np.array(data)
# labels = np.array(labels)

# 嵌入向量的目标形状是 (1000, 768)
target_shape = (512, 768)

processed_data = []
for embedding in data:
    if embedding.shape[0] < target_shape[0]:
        # 如果嵌入向量较短，使用零填充
        padding = target_shape[0] - embedding.shape[0]
        # 使用 np.pad 在第一个维度上进行零填充
        embedding = np.pad(embedding, ((0, padding), (0, 0)), mode='constant')
    elif embedding.shape[0] > target_shape[0]:
        # 如果嵌入向量较长，进行截断
        embedding = embedding[:target_shape[0], :]
    
    processed_data.append(embedding)

# 现在可以使用 np.stack() 将处理后的数据转换为 NumPy 数组
data = np.stack(processed_data)
labels = np.array(labels)

# Split data into training, validation, and test sets
train_size = int(0.6 * len(data))
val_size = int(0.2 * len(data))
test_size = len(data) - train_size - val_size

data_train, data_val, data_test = np.split(data, [train_size, train_size + val_size])
label_train, label_val, label_test = np.split(labels, [train_size, train_size + val_size])

print("Data loaded and split into train, validation, and test sets.")

# 将数据转换为 tensor，但避免一次性将整个数据集加载到内存中
data_train_tensor = torch.from_numpy(data_train).float()
label_train_tensor = torch.from_numpy(label_train).float()

data_val_tensor = torch.from_numpy(data_val).float()
label_val_tensor = torch.from_numpy(label_val).float()

data_test_tensor = torch.from_numpy(data_test).float()
label_test_tensor = torch.from_numpy(label_test).float()

# 创建 TensorDataset
train_dataset = TensorDataset(data_train_tensor, label_train_tensor)
val_dataset = TensorDataset(data_val_tensor, label_val_tensor)
test_dataset = TensorDataset(data_test_tensor, label_test_tensor)

# 创建 DataLoader，并设置 pin_memory 和更小的 batch size
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)


# Define the model
class CharPlantCNN(nn.Module):
    def __init__(self):
        super(CharPlantCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=args.nb_filter1, kernel_size=args.filter_len1, padding=args.filter_len1 // 2)
        self.conv2 = nn.Conv1d(in_channels=args.nb_filter1, out_channels=args.nb_filter2, kernel_size=args.filter_len2, padding=args.filter_len2 // 2)

        # self.conv1 = nn.Conv1d(in_channels=4, out_channels=args.nb_filter1, kernel_size=args.filter_len1, padding='same')
        # self.conv2 = nn.Conv1d(in_channels=args.nb_filter1, out_channels=args.nb_filter2, kernel_size=args.filter_len2, padding='same')
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(in_features=args.nb_filter2 * 512, out_features=args.hidden)
        self.fc2 = nn.Linear(in_features=args.hidden, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 调整输入维度
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


model = CharPlantCNN().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=1e-5)

# Training and validation
best_val_loss = float('inf')
patience_counter = 0

print("Starting training...")
start_time = time.time()

# For tracking losses
train_losses = []
val_losses = []
accuracies = []
val_accuracies = []

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # 在数据加载后检查并调整输入形状
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = torch.round(outputs.squeeze())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    accuracy = correct / total
    accuracies.append(accuracy)
    print(f'Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')


end_time = time.time()
print(f"Training completed in {int(end_time - start_time)} seconds.")


