# mypy: disable-error-code="import-not-found"
"""Defines a dummy "letters" which is solvable by the one-to-many model.

This model takes a random sequence of letters and outputs a new sequence
containing the unique letters repeated N times. For example, the input sequence
"abbbccdef" would be transformed into "aabbccddeeff".
"""

import argparse
import itertools
import random
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset

from monotonic_attention.one_to_many import OneToManyMultiheadMonotonicAttention

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    raise ModuleNotFoundError("Visualization requires matplotlib: `pip install matplotlib`")


class LettersDataset(IterableDataset[tuple[Tensor, Tensor]]):
    def __init__(self, num_letters: int, seq_length: int) -> None:
        super().__init__()

        assert 2 <= num_letters <= 26, f"`{num_letters=}` must be between 2 and 26"

        self.num_letters = num_letters
        self.seq_length = seq_length
        self.padding_idx = 0

        self.vocab = list("abcdefghijklmnopqrstuvwxyz"[:num_letters])

    def tokenize(self, s: str) -> Tensor:
        return Tensor([self.vocab.index(c) + 1 for c in s])

    def detokenize(self, t: Tensor) -> str:
        return "".join(self.vocab[int(i) - 1] for i in t.tolist())

    @property
    def vocab_size(self) -> int:
        return self.num_letters + 1

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        tokens_in: list[int] = []
        tokens_out: list[int] = []
        prev_letter: int | None = None
        while len(tokens_in) < self.seq_length:
            choices = [i for i in range(1, self.num_letters + 1) if i != prev_letter]
            letter = random.choice(choices)
            prev_letter = letter
            tokens_in.extend([letter] * min(self.seq_length - len(tokens_in), random.randint(2, 15)))
            tokens_out.extend([letter])

        tokens_in_t = torch.tensor(tokens_in)
        tokens_out_t = torch.tensor(tokens_out)
        return tokens_in_t, tokens_out_t

    def collate_fn(self, items: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        tokens_in, tokens_out = zip(*items)

        # Pads the output tokens and creates a mask.
        max_out_len = max(len(t) for t in tokens_out)
        tokens_out_t = torch.full((len(tokens_out), max_out_len), fill_value=self.padding_idx, dtype=torch.long)
        for i, token_out in enumerate(tokens_out):
            tokens_out_t[i, : len(token_out)] = token_out

        return torch.stack(tokens_in), tokens_out_t


class MonotonicSeq2Seq(nn.Module):
    """Defines a monotonic sequence-to-sequence model.

    Parameters:
        vocab_size: The vocabulary size
        dim: The number of embedding dimensions
    """

    def __init__(self, vocab_size: int, dim: int, padding_idx: int, use_rnn: bool) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embs = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.init_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.rnn = nn.LSTM(dim, dim, batch_first=True) if use_rnn else None
        self.attn = OneToManyMultiheadMonotonicAttention("many_keys_one_query", dim, num_heads=1)
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        bsz = tgt.size(0)
        src_emb = self.embs(src)
        tgt_emb = torch.cat((self.init_emb.expand(bsz, -1, -1), self.embs(tgt[..., :-1])), dim=1)
        x, _ = self.attn(tgt_emb, src_emb, src_emb)
        if self.rnn is not None:
            x, _ = self.rnn(x)
        x = self.proj(x)
        return x

    def get_attention_matrix(self, src: Tensor, tgt: Tensor) -> Tensor:
        bsz = tgt.size(0)
        src_emb = self.embs(src)
        tgt_emb = torch.cat((self.init_emb.expand(bsz, -1, -1), self.embs(tgt[..., :-1])), dim=1)
        return self.attn.get_attn_matrix(tgt_emb, src_emb)


def train(
    num_letters: int,
    seq_length: int,
    batch_size: int,
    device_type: str,
    embedding_dims: int,
    max_steps: int,
    save_path: str | None,
    use_rnn: bool,
) -> None:
    device = torch.device(device_type)

    ds = LettersDataset(num_letters, seq_length)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=ds.collate_fn)
    pad = ds.padding_idx

    model = MonotonicSeq2Seq(ds.vocab_size, embedding_dims, pad, use_rnn)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)

    for i, (tokens_in, tokens_out) in itertools.islice(enumerate(dl), max_steps):
        opt.zero_grad(set_to_none=True)
        tokens_in, tokens_out = tokens_in.to(device), tokens_out.to(device)
        tokens_out_pred = model(tokens_in, tokens_out)
        loss = F.cross_entropy(tokens_out_pred.view(-1, ds.vocab_size), tokens_out.view(-1), ignore_index=pad)
        loss.backward()
        opt.step()
        print(f"{i}: {loss.item()}")

    # Gets the attention matrix.
    tokens_in, tokens_out = next(ds)
    tokens_in = tokens_in.unsqueeze(0).to(device)
    tokens_out = tokens_out.unsqueeze(0).to(device)
    attn_matrix = model.get_attention_matrix(tokens_in, tokens_out)
    attn_matrix = attn_matrix[0, 0, 0].detach().exp().cpu().numpy()

    # Visualize the attention matrix against the letters.
    letters_in = ds.detokenize(tokens_in[0])
    letters_out = ds.detokenize(tokens_out[0])
    plt.figure()
    plt.imshow(attn_matrix, cmap="gray")
    plt.xticks(range(len(letters_in)), letters_in)
    plt.yticks(range(len(letters_out)), letters_out)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Attention")
    plt.colorbar()

    # Grid between adjacent cells.
    for i in range(len(letters_in)):
        plt.axvline(i, color="white", linewidth=0.5)
    for i in range(len(letters_out)):
        plt.axhline(i, color="white", linewidth=0.5)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def main() -> None:
    random.seed(1337)
    torch.manual_seed(1337)
    np.random.seed(1337)

    parser = argparse.ArgumentParser(description="Train a dummy letters model.")
    parser.add_argument("-n", "--num-letters", type=int, default=10, help="How many unique letters to use")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="The batch size to use")
    parser.add_argument("-s", "--seq-length", type=int, default=64, help="Input sequence length")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="The device to use for training")
    parser.add_argument("-e", "--embedding-dims", type=int, default=32, help="Number of embedding dimensions")
    parser.add_argument("-m", "--max-steps", type=int, default=100, help="Maximum number of steps to train for")
    parser.add_argument("-p", "--save-path", type=str, default=None, help="Where to save the visualized attentions")
    parser.add_argument("-u", "--use-rnn", action="store_true", help="Whether to use an RNN")
    args = parser.parse_args()

    train(
        num_letters=args.num_letters,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device_type=args.device,
        embedding_dims=args.embedding_dims,
        max_steps=args.max_steps,
        save_path=args.save_path,
        use_rnn=args.use_rnn,
    )


if __name__ == "__main__":
    # python -m examples.one_to_many_letters
    main()
