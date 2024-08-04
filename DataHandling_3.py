import torch
import numpy as np
from tokenizers import Tokenizer

# Özel tokenizatörü yükle
tokenizer = Tokenizer.from_file("Tokenizers/Nutuk_tokenizer.json")

vocab_size = tokenizer.get_vocab_size()
print(f"Token sayısı: {vocab_size}")

# Encode fonksiyonunu tanımla
def encode(s):
    return tokenizer.encode(s).ids

# Decode fonksiyonunu tanımla
def decode(l):
    return tokenizer.decode(l)

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split

        with open('Data_txt/NUTUK_updated2.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokens = encode(text)
        self.tokens = torch.tensor(tokens)
        
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # Split dataset into train and val sets
        split_idx = int(0.9 * len(self.tokens))  # 90-10 split for train-val
        if split == "train":
            self.tokens = self.tokens[:split_idx]
        elif split == "val":
            self.tokens = self.tokens[split_idx:]

        # Initial shard and position
        self.current_position = B * T * self.process_rank
        self.current_shard = 0
        
        # Split dataset into shards for distributed training
        self.shards = self.split_dataset()

    def split_dataset(self):
        total_length = len(self.tokens)
        shard_size = total_length // self.num_processes
        return [self.tokens[i*shard_size:(i+1)*shard_size] for i in range(self.num_processes)]

    def reset(self):
        self.current_shard = 0
        self.current_position = self.B * self.T * self.process_rank
        self.tokens = self.shards[self.current_shard]

    def next_batch(self):
        B, T = self.B, self.T
        while True:
            if self.current_position + B * T + 1 > len(self.tokens):
                # Move to the next shard if not enough tokens for a full batch
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = self.shards[self.current_shard]
                self.current_position = B * T * self.process_rank

            buf = self.tokens[self.current_position:self.current_position + B * T + 1]
            if buf.size(0) < B * T + 1:
                # If the current shard can't form a batch, restart from the beginning
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = self.shards[self.current_shard]
                self.current_position = B * T * self.process_rank
                continue

            x = buf[:-1].view(B, T)  # inputs
            y = buf[1:].view(B, T)  # targets

            # Advance the position in the tensor
            self.current_position += B * T * self.num_processes

            return x, y