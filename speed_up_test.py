from transformers import AutoModel, AutoTokenizer
from src.BranchyModelConfig import BranchyModelConfig
from src.BranchyModel import BranchyCausalModel
from logging import getLogger

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_penalty", type=float, default=0.9)
args = parser.parse_args()
model_penalty = args.model_penalty

logger = getLogger()
logger.setLevel("INFO")

device = "cuda:1"
model_conf = BranchyModelConfig.from_pretrained(f"valcore/branchy_phi-2_{model_penalty}")
model_conf.device = device
model = BranchyCausalModel(model_conf).to(device)
pretrained_backbone = torch.load("base_model.bin", map_location=device)
pretrained_head = torch.load("head_model_phi2_penaltyv2_0.9.bin", map_location=device)
model.model.load_state_dict({**pretrained_backbone, **pretrained_head})

context_length = 100

input_text = ["This story is", "I am a", "This script is", "Mercury "]
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to("cuda:1")
full_sentence = input_ids.clone().cpu()
head_count = torch.zeros(5, dtype=torch.int32)
for i in range(1000):
    output = model(input_ids)
    head_count += torch.bincount(output.head_indices, minlength=5).cpu()
    if i % 50 == 0:
        print(f"Step {i} : head count {head_count}, input_ids {input_ids.shape}")
    next_token_logits = output.logits[:, -1, :]
    next_token_id = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1).squeeze()
    full_sentence = torch.cat([full_sentence, next_token_id.unsqueeze(-1).cpu()], dim=-1)
    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
    if input_ids.shape[-1] > context_length:
        input_ids = input_ids[:, -context_length:]
for sentence in full_sentence:
    print(tokenizer.decode(sentence))
with open(f"head_usage_phi-2_{model_penalty}.txt", "w") as f:
    f.write(str(head_count.tolist()))
del model
torch.cuda.empty_cache()