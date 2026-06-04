from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")

sequence = "ATCGATCGATCG..."

encoding = tokenizer(
sequence,
return_tensors="pt",
return_offsets_mapping=True,
truncation=True,
max_length=512
)

offsets = encoding.pop("offset_mapping")

with torch.no_grad():
outputs = model(**encoding)

# token-level embedding

token_embeddings = outputs.last_hidden_state[0].cpu().numpy()

# sequence length

seq_len = len(sequence)

# initialize bp-level matrix

bp_embeddings = np.zeros((seq_len, 768))
bp_counts = np.zeros(seq_len)

offsets = offsets[0].tolist()

for i, (start, end) in enumerate(offsets):

```
# skip CLS/SEP
if start == end:
    continue

bp_embeddings[start:end] += token_embeddings[i]
bp_counts[start:end] += 1
```

# avoid division by zero

bp_counts[bp_counts == 0] = 1

# average overlapping regions

bp_embeddings = bp_embeddings / bp_counts[:, None]

# save bp-level embeddings

np.save("bp_embedding.npy", bp_embeddings)
