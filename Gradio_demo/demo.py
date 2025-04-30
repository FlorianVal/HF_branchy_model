import gradio as gr
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "valcore/Branchy-Phi-2" 
tokenizer_name = "microsoft/phi-2"  

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model.eval()

def generate_next_token(model, tokenizer, text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model(inputs)

    next_token_logits = outputs.logits[:, -1, :].squeeze()
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    next_token_id = torch.argmax(next_token_probs, dim=-1)
    next_token = tokenizer.decode(next_token_id, return_tensors="pt")

    head_index = outputs.head_indices
    print(head_index)
    return next_token, head_index

import matplotlib.pyplot as plt

def generate_text(prompt, early_exit=False):
    if early_exit:
        model.head_thresholds = [2.0270779132843018, 1.8969502449035645, 1.4789371490478516, 0.9875392913818359]
    else:
        model.head_thresholds = [10., 10., 10., 10.]

    output = prompt
    head_indices = []

    for i in range(50):  # generate up to 50 tokens
        next_token, head_index = generate_next_token(model, tokenizer, output)
        output += next_token
        head_indices.append(head_index)
        
        update = gr.LinePlot(value=pd.DataFrame({"indices": head_indices, "x": range(len(head_indices))}), x="x", y="indices")

        yield output, update


        if next_token == tokenizer.eos_token:
            break

demo = gr.Interface(
    fn=generate_text,
    inputs=["text", "checkbox"],
    outputs=["text", "lineplot"],
    title="Text Generation with Head Index",
    description="Enter a prompt and see the model output, the head index for each token, and a live plot of the head indices.",
    examples=[["Once upon a time,"], ["The quick brown fox jumps over the lazy dog."]],
    allow_flagging="never",
    live=True,
)



demo.launch()
