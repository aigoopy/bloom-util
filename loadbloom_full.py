#For layer_number error: pip install transformers==4.20.0
import os
import datetime
import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm
import psutil

def get_state_dict(shard_num, prefix=None):
    d = torch.load(os.path.join(model_path, f"pytorch_model_{shard_num:05d}-of-00072.bin"))
    return d if prefix is None else OrderedDict((k.replace(prefix, ''), v) for k, v in d.items())

from transformers import AutoTokenizer, AutoModelForCausalLM, BloomConfig
from transformers.models.bloom.modeling_bloom import BloomBlock, build_alibi_tensor

model_path = "/repos/bloom" # replace with your local folder path
config = BloomConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = 'cpu'

def load_embeddings():
    state_dict = get_state_dict(shard_num=1, prefix="word_embeddings_layernorm.")
    embeddings = nn.Embedding.from_pretrained(state_dict.pop('word_embeddings.weight'))
    lnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, dtype=torch.bfloat16)
    lnorm.load_state_dict(state_dict)
    return embeddings.to(device), lnorm.to(device)

def load_causal_lm_head():
    linear = nn.utils.skip_init(
        nn.Linear, config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16)
    linear.load_state_dict(get_state_dict(shard_num=1, prefix="word_embeddings."), strict=False)
    return linear.bfloat16().to(device)

def load_block(block_num):
    block_obj = BloomBlock(config, layer_number=block_num).bfloat16()
    block_obj.load_state_dict(get_state_dict(shard_num=block_num + 2, prefix=f"h.{block_num}."))
    return block_obj

def load_final_lnorm():
    final_lnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, dtype=torch.bfloat16)
    final_lnorm.load_state_dict(get_state_dict(shard_num=72, prefix="ln_f."))
    final_lnorm.to(device)
    return final_lnorm

#RAM Check
mem_req = 378493992960 # Total BLOOM model size
mem_avail = psutil.virtual_memory().available
print("Required  Memory: " + str(mem_req))
print("Available Memory: " + str(mem_avail))
if mem_req > mem_avail:
    print("Not enough memory.")
    exit()
else:
    print("Remaining Memory: " + str(round((mem_avail-mem_req) / 1024 / 1024 / 1024, 2)) + " GB")

# Load all modules to RAM and GPU (except the blocks, which are only loaded to RAM)
block_count = 70
block_num = 0
blocks = [None] * block_count
print(str(datetime.datetime.now()) + "-Started Block Loading")

for block_num in tqdm(range(block_count)):
        blocks[block_num] = load_block(block_num)
        #print(str(datetime.datetime.now()) +  "-Loaded block: " + str(block_num) + "/" + str(block_count))

#blocks = [load_block(block_num) for block_num in range(70)]
embeddings, emb_lnorm = load_embeddings()
print(str(datetime.datetime.now()) +  "-Finished load_embeddings")

lm_head = load_causal_lm_head()
print(str(datetime.datetime.now()) +  "-Finished load_causal_lm_head")

final_lnorm = load_final_lnorm()
print(str(datetime.datetime.now()) +  "-Finished load_final_lnorm")

def forward(input_ids):
    # 1. Create attention mask and position encodings
    attention_mask = torch.ones(len(input_ids)).unsqueeze(0).bfloat16().to(device)
    alibi = build_alibi_tensor(input_ids.shape[1], config.num_attention_heads,
                               torch.bfloat16).to(device)
    # 2. Use word embeddings and associated lnorm
    hidden_states = emb_lnorm(embeddings(input_ids))

    # 3. Use the BLOOM blocks sequentially
    for block in blocks:
        block_gpu = block.to(device) # Move single block to GPU
        hidden_states = block_gpu(hidden_states, attention_mask=attention_mask, alibi=alibi)[0]
    
    hidden_states = final_lnorm(hidden_states)
    
    #4. Use language model head
    logits = lm_head(hidden_states)

    # 5. Compute next token 
    return torch.argmax(logits[:, -1, :], dim=-1)

while True:   
    input_sentence = str(input("Enter prompt for inference: "))
    max_tokens = int(input("Enter maximum tokens: "))    
    input_ids = tokenizer.encode(input_sentence, return_tensors='pt').to(device)
    for i in range(max_tokens): 
        print(str(datetime.datetime.now()))
        print(f"Token {i + 1} ", end='')
        new_id = forward(input_ids)
        input_ids = torch.cat([input_ids, new_id.unsqueeze(-1)], dim=-1)
        print(tokenizer.decode(new_id))

    print(tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True))