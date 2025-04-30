
from collections import OrderedDict
from hamcrest import is_
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import copy

from dataclasses import dataclass 
from torch import Tensor
from .BranchyModelConfig import BranchyModelConfig
from typing import List, Optional, Dict, Tuple
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class Branch(nn.Module):
    """
    A branch module for use in the BranchyModel, representing an auxiliary output head attached at a specified layer
    within a transformer model. Each branch processes the output of its corresponding layer and produces an output
    which can be used for early exits or auxiliary tasks.

    This class is designed to be flexible, allowing for different configurations of the linear layer based on the
    underlying model's architecture.

    Attributes:
        layernorm (torch.nn.LayerNorm): Applies Layer Normalization over a mini-batch of inputs.
        lm_head (torch.nn.Linear): The linear layer that maps the hidden states to the vocabulary size, producing
            the output logits for each token in the sequence.

    Example Usage:
        # Assuming `config` is an instance of the model's configuration class with attributes `hidden_size` and
        # `vocab_size` properly set.
        branch = Branch(config)

        # `x` is a tensor representing the output from a transformer layer, shaped as [batch_size, seq_length, hidden_size]
        output_logits = branch(x)
    """
    def __init__(self, config: BranchyModelConfig):
        """
        Initializes the Branch module.

        Args:
            config (PretrainedConfig): The configuration object containing parameters like hidden size and vocabulary
            size. This object provides the necessary settings for initializing the layer normalization and linear
            layers within the Branch.
        """
        super().__init__()
        self.layernorm: nn.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head: nn.Linear = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Branch module.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_length, hidden_size], representing the output
            from a transformer layer.

        Returns:
            Tensor: Output logits of shape [batch_size, seq_length, vocab_size], resulting from passing the
            input through layer normalization and a linear layer.
        """
        x = self.layernorm(x)
        x = self.lm_head(x)
        return x


class BranchyCausalModel(PreTrainedModel):
    """A class for Causal branchy Model, this one integrate the early exit mechanism and only output one logit on each step as a conventional model.
    """
    config_class = BranchyModelConfig

    def __init__(self,
                 config: BranchyModelConfig):
        super().__init__(config)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_str)
        self.lm_head = self.model.lm_head
        self.vocab_size = self.model.vocab_size
        self.model = self.model.model
        self.head_thresholds = torch.tensor(config.head_thresholds) 
        self.confidence_metric_fn = breaking_ties
        
        # Get number of layer from main model
        if hasattr(self.model.config, "n_layer") or hasattr(self.model.config, "num_hidden_layers"): 
            self.num_layers = (
                self.model.config.n_layer
                if hasattr(self.model.config, "n_layer")
                else self.model.config.num_hidden_layers
            )
            assert self.num_layers is not None and self.num_layers > 0, "n_layer must be a positive integer."
        else:
            raise ValueError("cannot find n_layer in config")
        
        assert config.branch_number < self.num_layers , "branch_number must be a positive integer less than the number of layers in the model."
        
        # If we provide only the number of branches, we will distribute them evenly across the model
        if config.branch_locations is None:
            interval = self.num_layers // (config.branch_number + 1)   
            config.branch_locations = [i * interval for i in range(1, config.branch_number+1)]
        
        # Check that specified branch locations are within the range of the model's layers
        if any([loc >= self.num_layers for loc in config.branch_locations]):
            raise ValueError("Branch location exceeds the number of layers in the model.")
           
        self.branches = torch.nn.ModuleList()
        if config.copy_lm_head:
            logger.info("Fine-tuning branches")
            for branch in config.branch_locations:
                self.branches.append(copy.deepcopy(self.lm_head))
        else:
            for _ in config.branch_locations:
                new_branch = Branch(self.model.config)
                new_branch.apply(self.model._init_weights)
                self.branches.append(new_branch)
        
        self.gradient_checkpointing = False
        self.post_init()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        self.head_thresholds = self.head_thresholds.to(*args, **kwargs)
        return self
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        
        return model_inputs
    
    def model_pre_forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.model.config.use_cache

        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if self.model.gradient_checkpointing and self.model.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        use_legacy_cache = None
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
            
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        inputs_embeds = self.model.embed_dropout(inputs_embeds)

        # Attention mask.
        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        return inputs_embeds, use_legacy_cache, attention_mask, position_ids, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict
    
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                head_window_size: Optional[int] = None,
                ):
        use_cache = False # Disable it for now TODO Update how cache is handled to allow early exits
        inputs_embeds, use_legacy_cache, attention_mask, position_ids, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict = self.model_pre_forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)

        
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_logits = ()
        is_early_exited = False
        next_decoder_cache = None
        
        for layer, decoder_layer in enumerate(self.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.model.gradient_checkpointing and self.model.training:
                layer_outputs, use_legacy_cache = self.model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                )
                hidden_states = layer_outputs[0]
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = layer_outputs[0]
                if layer in self.config.branch_locations:
                    logits = self.branches[self.config.branch_locations.index(layer)](layer_outputs[0])
                    if not self.training:
                        # During inference, calculate score on the fly to decide if we should early exit
                        score = self.confidence_metric_fn(logits)[..., -1] # score for the classified token TODO migth be interesting to take score from whole vector ?
                        if score > self.head_thresholds[self.config.branch_locations.index(layer)]:
                            is_early_exited = True
                            logger.debug(f"Early exit at layer {layer} with score {score}")
                            break
                    else:
                        # if in training we return full logits
                        all_logits += (logits,)

            

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        if not is_early_exited:
            logger.debug(f"No early exit")
            hidden_states = self.model.final_layernorm(hidden_states)
            logits = self.lm_head(hidden_states)
            logits = logits.float()
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        loss = [None, None, None, None]   
        if self.training:
            loss = self.compute_self_supervision_loss(
                torch.stack(all_logits), hidden_states
            )
        if not return_dict:
            raise NotImplementedError("return_dict=False is not implemented")
        return CausalBranchyLLMOutputWithPast(
            loss=loss[0],
            head_loss=loss[1],
            entropies=loss[2],
            entropy=loss[3],
            logits=logits, # shape (batch_size, seq_len, vocab_size)
            head_logits=all_logits, # shape (num_branches, batch_size, seq_len, vocab_size)
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            head_indices=layer,
        )
        
    def compute_self_supervision_loss(self,
                                      aux_logits: torch.Tensor,
                                      lm_logits: torch.Tensor,
                                      return_dict: bool = True
                                      ) -> Dict[str, torch.Tensor]:
        
        last_aux_logits = aux_logits[..., -1, :]
        last_lm_logits = lm_logits[..., -1, :]

        losses = ()
        entropies = ()
        # Can be useful to have detailed loss per head for comparison of performance
        for head_logit in last_aux_logits:
            ce_loss = nn.CrossEntropyLoss(reduction="mean")(
                head_logit, torch.argmax(last_lm_logits, dim=-1)
            )
            probas = F.softmax(head_logit, dim=-1)
            log_probas = torch.log(probas + 1e-8)
            assert not torch.isnan(log_probas).any(), "NaNs found in log_probas"
            entropy = -torch.sum(probas * log_probas, dim=-1)
            assert not torch.isnan(entropy).any(), "NaNs found in entropy before mean"
            entropy = torch.mean(entropy)
            entropies += (entropy,)
            losses += ((1 - self.config.penalty_weight) * ce_loss - self.config.penalty_weight * entropy,)
            
        loss = torch.stack(losses, dim=0).mean(dim=-1)
        entropy = torch.stack(entropies, dim=0).mean(dim=-1)
        if not return_dict:
            return tuple(v for v in (loss, losses, entropy, entropies) if v is not None)
        return SelfSupervisedLossOutput(
                loss=loss,
                head_losses= losses,
                entropies= entropies,
                entropy= entropy
        )
        
@dataclass
class CausalBranchyLLMOutputWithPast(ModelOutput):
    loss: Optional[torch.Tensor] = None # Main loss
    head_loss: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    entropies: Optional[Tuple[torch.Tensor]] = None
    logits: torch.Tensor = None
    head_logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    head_indices: Optional[torch.Tensor] = None
    
@dataclass
class SelfSupervisedLossOutput(ModelOutput):
    loss: torch.Tensor = None
    head_losses: torch.Tensor = None
    entropy: torch.Tensor = None
    entropies: torch.Tensor = None