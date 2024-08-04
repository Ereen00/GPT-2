from dataclasses import dataclass
import DataHandling_3

with open('Data_txt/wp_chat_history_emogies.txt', 'r', encoding='utf-8') as f:
    textSample = f.read()

eval_interval = 20 # kaç adımda bir değerlendirme yapılacak

B = 2 # micro batch size
T = 1024 # sequence length
total_batch_size = B * T * 1 # (1 = ddp_world_size)

epoch = len(DataHandling_3.encode(textSample)) // (B*T)

max_lr = 3e-4
min_lr = max_lr * 0.1
max_steps = epoch * 4
warmup_steps = max_steps * 0.05

print(f"warm up steps: {int(warmup_steps)}")

max_length = 32 
top_k = 50
temperature = 1.0

@dataclass
class GPTConfig:
    block_size: int = T # max sequence length
    vocab_size: int = DataHandling_3.vocab_size 
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension 12*64