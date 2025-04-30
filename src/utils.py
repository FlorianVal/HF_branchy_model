from logging import config

from requests import head
from sympy import use
import tqdm
from .BranchyModel import BranchyModel
from .BranchyModelConfig import BranchyModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import logging
import torch
import os
import glob
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_default_branchy_model(model_str: str = "susnato/phi-1_5_dev",
                               branch_locations: List[int] = None,
                               branch_number: int = 3,
                               penalty_weight: Optional[float] = 1.,
                               head_window_size: int = 512,):
    """
    Load a default BranchyModel with a specified model string.
    
    Args:
        model_str (str): The model string to load. Defaults to "susnato/phi-1_5_dev".
        
    Returns:
        BranchyModel: The default BranchyModel loaded with the specified model string.
        Tokenizer: The tokenizer for the specified model string.
    """
    model = AutoModelForCausalLM.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    num_layers = model.config.num_hidden_layers
    logger.info(f"Loaded model: {model_str} with {num_layers} layers.")

    
    config = BranchyModelConfig(branch_locations=branch_locations,
                                penalty_weight=penalty_weight,
                                head_window_size=head_window_size,
                                branch_number=branch_number)
    
    logger.info(f"Loding BranchyModel with config: {config}")
    
    return BranchyModel(config, model), tokenizer 

def load_model_from_ckpt(model: BranchyModel,
                         ckpt_path: str,
                         base_model_path: Optional[str] = "base_model.bin",
                         device: Optional[str] = "cpu"):
    """
    Load a model from a checkpoint file.
    
    Args:
        model (BranchyModel): The model to load the checkpoint into.
        ckpt_path (str): The path to the checkpoint file.
    """
    base_model_dict = torch.load(base_model_path, map_location=device)
    heads_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict({**base_model_dict, **heads_dict})
    model = model.to(device)
    return model

def breaking_ties(tensor: torch.Tensor):
    """
    Break ties in a tensor by subtracting the second highest value from the highest value.
    
    Args:
        tensor (torch.Tensor): The tensor to break ties in. shape [..., vocab_size]
    
    Returns:
        torch.Tensor: The tensor with ties broken. shape [...]
        
    Example: 
    Input : Tensor of shape [head_number, batch, seq_len, vocab_size]
    Output: Tensor of shape [head_number, batch, seq_len]
    """
    return torch.sub(torch.topk(tensor, 2, dim=-1).values[..., 0], torch.topk(tensor, 2, dim=-1).values[..., 1])


def compute_calibration_set(model: BranchyModel,
                            tokenizer: AutoTokenizer,
                            starting_batch: List[str] = None,
                            metric: str = "max",
                            size: int = 100,
                            max_seq_length: int = 512,
                            device="cpu"):
    """
    Do inference on a BranchyModel to compute the calibration set.
    
    Args:
        model (BranchyModel): The model to use for inference.
        starting_batch (List[str]): The starting batch of text to use for inference.
        metric (str): The metric to use for selecting the calibration set. Defaults to "max".
        size (int): The size of the calibration set to compute. Defaults to 100.
    
    Returns:
        calibration (torch.Tensor): The calibration set.  
        truth (torch.Tensor): The truth values for the calibration set. Comparing the output of the main model to the output of the heads.
    """
    calibration = torch.Tensor([])
    truth = torch.BoolTensor([])
    inputs_ids = tokenizer(starting_batch, return_tensors="pt").input_ids.to(model.device)
    progress_bar = tqdm.tqdm(range(size))
    for _ in progress_bar: 
        out = model(inputs_ids)
        inputs_ids = append_batch(inputs_ids, out, max_seq_length)
        #print(inputs_ids.shape)
        logits = out.logits.detach().cpu()
        head_logits = out.head_logits.detach().cpu()
        if metric == "max":
            calibration = torch.cat((calibration, torch.max(head_logits, -1).values[...,-1].T), 0) # of shape [batch_size, head_number]
        if metric == "bt":
            calibration = torch.cat((calibration, breaking_ties(head_logits)[..., -1].T), 0)
        heads_pred = torch.argmax(head_logits, -1)[..., -1]
        lm_preds = torch.argmax(logits, -1)[..., -1].unsqueeze(0).repeat((head_logits.shape[0]),1)
        truth = torch.cat((truth, (heads_pred == lm_preds).T), 0)
    logger.info(f"Generated content for calibration : {tokenizer.batch_decode(inputs_ids)}")
    return calibration, truth # Of shape [batch * size, heads]

def append_batch(batch, outputs, max_length):
    """Append the outputs of the model to the batch and truncate if necessary

    Args:
        batch (torch.Tensor): Batch of sequences
        outputs (CausalBranchyLLMOutputWithPast): Dictionary of outputs from the model
        max_length (int): Maximum length of the sequence
        
    Returns:
        torch.Tensor: New batch of sequences
    """
    batch = torch.cat(
        (
            batch,
            torch.multinomial(
                torch.softmax(outputs.logits[:, -1, :], dim=-1), 1
            ),
        ),
        dim=-1,
    )
    # truncate batch if it is too long
    if batch.shape[1] > max_length:
        batch = batch[:, -max_length :]
    return batch

def compare_tensors(id_tensor, tensor_to_sort):
    # assume both tensors have are of same shape ([head_number, batch_size])
    
    gt = id_tensor[-1] # assume head is the last dim
    head_ids = id_tensor[:-1]
    tensor_to_sort = tensor_to_sort[:-1]
    #print(f"Ground truth: {gt.shape}, Head ids: {head_ids.shape}")
    bool_tens = (head_ids == gt.unsqueeze(0).repeat(head_ids.shape[0], 1)).float()
    #print(f"Bool tensor: {bool_tens.shape}")
    val, indexes = torch.sort(tensor_to_sort, dim=1, descending=True)
    #print(f"Indexes: {indexes.shape}")
    sorted_bool_tens = torch.gather(bool_tens, 1, indexes)

    return sorted_bool_tens


def visualize_comparison_3d(tensor_3d):
    n_models = tensor_3d.shape[0]
    plt.figure(figsize=(n_models * tensor_3d.shape[2] * 0.15, tensor_3d.shape[1] * 0.5))
    
    for i in range(n_models):
        plt.subplot(1, n_models, i+1)
        plt.imshow(tensor_3d[i].cpu().numpy(), cmap='RdYlGn', aspect='equal', interpolation='nearest')
        plt.axis('off')
        plt.title(f"{i+1}")
    
    # save
    os.makedirs("results", exist_ok=True)
    existing_files = os.listdir("results")
    file_name = f"results/comparison_{len(existing_files)}.png"
    plt.savefig(file_name)
    plt.show()


# Accumulation des résultats dans un tensor 3D
def accumulate_comparisons(model_str, branch_numbers=3, branch_locations=None, iterations=100, initial_seq="This is a story about", max_num_model=None, base_path="/app/head_model_phi2*.bin",device=None):
    accum_max_tensor, accum_bt_tensor, accum_entropy_tensor = [None for _ in range(3)]
    file_dict = {}
    for path in glob.glob(base_path):
        # Extract the penalty number from the file path
        penalty_value = path.split("penalty_")[-1].split(".bin")[0]
        file_dict[penalty_value] = path
    for num_model, ckpt_path in enumerate(sorted(file_dict.values())):
        branchy_model = None
        print(f"Loading model from {ckpt_path}")
        branchy_model, tokenizer = load_default_branchy_model(model_str, branch_locations, branch_numbers)
        branchy_model = load_model_from_ckpt(branchy_model, ckpt_path, base_model_path="/app/base_model.bin")
        # print hash of the model to check if it is the same
        print(f"Model hash: {hash(branchy_model)}")
        print(branchy_model.branches[0].lm_head.weight[0, :10])
        if device is not None:
            branchy_model.to(device)
        max_tensor, bt_tensor, entropy_tensor, id_tensor = generate_and_analyze_text(branchy_model, tokenizer, initial_seq, iterations=iterations, device=device)
        
        sorted_bool_tens_max = compare_tensors(id_tensor, max_tensor) # shape [head_number, batch_size]
        sorted_bool_tens_bt = compare_tensors(id_tensor, bt_tensor)
        sorted_bool_tens_entropy = compare_tensors(id_tensor, entropy_tensor)

        # Empiler ou initialiser accum_tensor
        if accum_max_tensor is None:
            accum_max_tensor = sorted_bool_tens_max.unsqueeze(0)
            accum_bt_tensor = sorted_bool_tens_bt.unsqueeze(0)
            accum_entropy_tensor = sorted_bool_tens_entropy.unsqueeze(0)
        else:
            accum_max_tensor = torch.cat([accum_max_tensor, sorted_bool_tens_max.unsqueeze(0)], dim=0)
            accum_bt_tensor = torch.cat([accum_bt_tensor, sorted_bool_tens_bt.unsqueeze(0)], dim=0)
            accum_entropy_tensor = torch.cat([accum_entropy_tensor, sorted_bool_tens_entropy.unsqueeze(0)], dim=0)
        if max_num_model is not None and num_model >= max_num_model:
            break
    return accum_max_tensor, accum_bt_tensor, accum_entropy_tensor

def generate_and_analyze_text(branchy_model, tokenizer, initial_seq, iterations=100, max_length=None, padding=True, device=None):
    """
    Generates and analyzes text using the provided branchy_model and tokenizer.

    Args:
        branchy_model (model): The branchy model used for text generation.
        tokenizer (tokenizer): The tokenizer used to encode and decode text.
        initial_seq (list): The initial sequence of tokens to start the generation.
        iterations (int, optional): The number of iterations for text generation. Defaults to 100.
        max_length (int, optional): The maximum length of the generated sequence. Defaults to None.

    Returns:
        tuple: A tuple containing the following tensors:
            - max_tensor: A tensor containing the maximum values of the logits for each generated token.
            - bt_tensor: A tensor containing the breaking ties values for each generated token.
            - entropy_tensor: A tensor containing the entropy values for each generated token.
            - id_tensor: A tensor containing the indices of the maximum values for each generated token.
    """    
    if padding:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        
    new_seq = tokenizer(initial_seq, return_tensors="pt", padding=padding).input_ids
    if device is not None:
        new_seq = new_seq.to(device)
    max_tensor, bt_tensor, entropy_tensor, id_tensor = [torch.tensor([]).to("cpu") for _ in range(4)]
    logger.info(f"Generating text with {iterations} iterations.")
    pbar = tqdm.tqdm(range(iterations))
    for _ in pbar:
        generation_out = generate_next_token(branchy_model, new_seq, method='greedy')  
        head_logits = generation_out["head_logits"].detach().cpu() # shape [head_number, batch_size, vocab_size]
        logits = generation_out["logits"].detach().cpu() # shape [batch_size, vocab_size]
        head_logits = torch.cat([head_logits, logits.unsqueeze(0)]) # shape [ head_number + 1, batch_size, vocab_size]
        max_tensor = torch.cat([max_tensor, torch.max(head_logits, dim=-1).values], dim=-1) # torch.max(head_logits, dim=-1).values of shape [head_number + 1, batch_size] max_tensor -> accumulate on batch_size dim
        bt_tensor = torch.cat([bt_tensor, breaking_ties(head_logits)], dim=-1) 
        id_tensor = torch.cat([id_tensor, torch.argmax(head_logits, dim=-1)], dim=-1) 
        softmaxed_logits = torch.nn.functional.softmax(head_logits, dim=-1)
        entropy_tensor = torch.cat([entropy_tensor, -torch.sum(softmaxed_logits * torch.log(softmaxed_logits + 1e-8), dim=-1)], dim=-1)
        new_seq = generation_out["new_seq"] # shape [batch_size, seq_len]
        if max_length is not None and new_seq.shape[1] > max_length:
            new_seq = new_seq[:, -max_length:]
    logger.info(f"Generated content: {tokenizer.batch_decode(new_seq)}")
    return max_tensor, bt_tensor, entropy_tensor, id_tensor

def generate_next_token(model, input, method='greedy'):
    """
    Generate the next token of a sequence using the given model and tokenizer.
    Specific for multi-branched models.
    Only output token from last head.

    Args:
        model (torch.nn.Module): The model to use for generation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for generation.
        input (str): The input text to generate from.
        method (str, optional): The generation method to use. Defaults to 'greedy'.
        padding (bool, optional): Whether to pad the input sequence. Defaults to True.

    Returns:
        dict: A dictionary containing the following outputs:
            - last_token (torch.Tensor): The next token in the sequence (from the last head).
            - logits (torch.Tensor): The logits of the next token, of shape [batch_size, vocab_size].
            - new_seq (str): The new sequence after adding the next token.
            - head_logits (torch.Tensor): The logits of the next token for each head, of shape [head_number, batch_size, vocab_size].
    """
    device = model.device

    # input_ids = tokenizer(input, return_tensors="pt", padding=padding).input_ids.to(device)
    model.eval()
    outputs = model(input) 
    head_logits = outputs.head_logits[..., -1, :] # of shape [head_number, batch_size, vocab_size]
    logits = outputs.logits[..., -1, :] # of shape [batch_size, vocab_size]
    if head_logits == []:
        raise ValueError("Model does not have head_outputs")
    if method == 'greedy':
        last_token = torch.argmax(logits, dim=-1)
    elif method == 'sample':
        last_token = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), num_samples=1)
    elif method == 'top_k':
        k = 5
        top_k = torch.topk(logits, k, dim=-1)
        top_k_logits, top_k_indices = top_k.values, top_k.indices
        top_k_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
        last_token = top_k_indices[torch.arange(top_k_probs.shape[0]), torch.multinomial(top_k_probs, num_samples=1).squeeze()]
    elif method == 'top_p':
        p = 0.9
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        tmp_logits = logits.clone()
        for i in range(logits.shape[0]):
            tmp_logits[i, indices_to_remove[i]] = float('-inf')
        last_token = torch.multinomial(torch.nn.functional.softmax(tmp_logits, dim=-1), num_samples=1).squeeze()
    else:
        raise ValueError(f"Unknown method: {method}")
    new_seq = torch.cat((input, last_token.unsqueeze(-1)), dim=-1).squeeze()
    
    return {
        "last_token": last_token,
        "logits": logits,
        "new_seq": new_seq,
        "head_logits": head_logits
    }
