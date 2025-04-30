# Gradio interface
from configparser import NoOptionError
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

#model_name = "valcore/Branchy-Phi-2"
#tokenizer_name = "microsoft/Phi-2"
#tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#model = AutoModelForCausalLM.from_pretrained(model_name)



# Function to generate text with early exit
def generate_with_early_exit(prompt, use_early_exit=True):
    
    return None

# Function to handle user input and display results
def process_prompt(prompt, use_early_exit):
    
    heads = []
    if use_early_exit:
        # Using Epsilon = 0.9
        model.head_thresholds = [3.187114715576172, 3.442272663116455, 2.636230945587158, 2.460529088973999]
    else:
        model.head_thresholds = [10., 10., 10., 10.]
        
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text with early exit
    start_time = time.time()
    outputs = model(input_ids)
    
    heads.append(outputs.head_indices)
    input_ids = torch.cat(input_ids, outputs.logits.argmax(-1)[:, -1:])
    
        
    print(prompt, use_early_exit)
    return "Hello !", 30.0


iface = gr.Interface(
    fn=process_prompt,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter a sentence prompt here..."),
        gr.Checkbox(label="Use Early Exit")
    ],
    outputs=[
        gr.Markdown(label="Generated Text with Early Exited Tokens Highlighted"),
        gr.Number(label="Processing Time (seconds)")
    ],
    title="Early Exit Language Model Demonstration",
    description="This app demonstrates how a language model with early exit works. Enter a sentence prompt to see the generated text and the tokens that were early exited."
)

# Launch the app
iface.launch()