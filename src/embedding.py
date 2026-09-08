from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
EMBED_DIM = 768
def load_dnabert2(
    model_name: str = MODEL_NAME,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """Load the DNABERT-2 tokenizer and encoder once for repeated inference."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model, torch.device(device)


def token_to_base_projection(
    sequence: str,
    token_embeddings: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Project token-level embeddings back to nucleotide coordinates.

    Parameters
    ----------
    sequence:
        Input nucleotide sequence.
    token_embeddings:
        Array of shape (n_tokens, 768), including any special tokens.
    offsets:
        Array of shape (n_tokens, 2) returned by the tokenizer.

    Returns
    -------
    np.ndarray
        Array of shape (len(sequence), 768), where each nucleotide position
        contains the mean of all token embeddings whose offset covers that
        nucleotide position.
    """
    sequence = sequence.upper()
    seq_len = len(sequence)

    if token_embeddings.ndim != 2 or token_embeddings.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Expected token embeddings with shape (n_tokens, {EMBED_DIM}), "
            f"got {token_embeddings.shape}"
        )
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError(f"Expected offsets with shape (n_tokens, 2), got {offsets.shape}")
    if len(offsets) != len(token_embeddings):
        raise ValueError("Number of offsets and token embeddings must match")

    bp_sum = np.zeros((seq_len, EMBED_DIM), dtype=np.float32)
    bp_count = np.zeros(seq_len, dtype=np.int32)

    for i, (start, end) in enumerate(offsets.tolist()):
        start, end = int(start), int(end)
        # CLS/SEP and other special tokens normally have start == end == 0.
        if start == end:
            continue
        if start < 0 or end > seq_len or start >= end:
            raise ValueError(f"Invalid token offset ({start}, {end}) for sequence length {seq_len}")
        bp_sum[start:end] += token_embeddings[i].astype(np.float32, copy=False)
        bp_count[start:end] += 1

    uncovered = np.flatnonzero(bp_count == 0)
    if uncovered.size:
        raise ValueError(
            "Token-to-base projection left uncovered nucleotide positions; "
            f"first positions: {uncovered[:10].tolist()}"
        )

    return bp_sum / bp_count[:, None].astype(np.float32)


def generate_base_embedding(
    sequence: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 512,
) -> np.ndarray:
    """Generate a 768-d representation aligned to every nucleotide."""
    sequence = sequence.upper()
    if not sequence:
        raise ValueError("Empty sequence")

    encoding = tokenizer(
        sequence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    offsets = encoding.pop("offset_mapping")[0].cpu().numpy()

    # A 500-bp input should be fully represented.  Do not silently truncate or
    # pad here; silent shape changes would invalidate nucleotide-level analysis.
    max_end = int(offsets[:, 1].max()) if offsets.size else 0
    if max_end != len(sequence):
        raise ValueError(
            f"Tokenizer output does not fully cover the sequence: max offset {max_end}, "
            f"sequence length {len(sequence)}. Increase max_length or inspect tokenization."
        )

    model_inputs = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**model_inputs)

    token_embeddings = outputs.last_hidden_state[0].detach().cpu().numpy()
    base_embedding = token_to_base_projection(sequence, token_embeddings, offsets)

    if base_embedding.shape != (len(sequence), EMBED_DIM):
        raise RuntimeError(
            f"Unexpected base embedding shape {base_embedding.shape}; "
            f"expected {(len(sequence), EMBED_DIM)}"
        )
    return base_embedding


def save_embedding(path: str | Path, embedding: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embedding.astype(np.float32, copy=False))


if __name__ == "__main__":
    # Small standalone smoke test.  The main preprocessing workflow imports
    # generate_base_embedding() from this module.
    import argparse

    parser = argparse.ArgumentParser(description="Generate a DNABERT-2 base-aligned embedding")
    parser.add_argument("sequence", help="DNA sequence (e.g. ACGT...)")
    parser.add_argument("--out", default="bp_embedding.npy")
    args = parser.parse_args()

    tok, mdl, dev = load_dnabert2()
    emb = generate_base_embedding(args.sequence, tok, mdl, dev)
    save_embedding(args.out, emb)
    print(f"Saved {emb.shape} embedding to {args.out}")
