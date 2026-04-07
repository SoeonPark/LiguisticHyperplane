# Code adapted from "https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb#scrollTo=Yx2tgILmXs2D"

import os
from IPython import get_ipython
ip = get_ipython()

# if not ip.extension_manager.loaded:
#     ip.extension_manager.load_extension('autoreload')
#     %autoreload 2
    
import circuitsvis as cv
# breakpoint()

import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px

from jaxtyping import Float
from functools import partial
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer, FactoredMatrix

torch.set_grad_enabled(False) # This code is for checking Inference, not training

# Plotting Helper Functions
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x": xaxis, "y": yaxis}, **kwargs).show(renderer)    
def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs).show(renderer)
def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(x=x, y=y, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
    
# Loading and Running Models
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# This is an sample text to test the model on, and to show how's the model's output and loss look like. 
model_description_text = """## Loading Models
HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. See my explainer for documentation of all supported models, and this table for hyper-parameters and the name used to load them. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 
For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""
loss = model(model_description_text, return_type="loss") 
print(f"Loss: {loss.item():.4f}")

# Caching all Activations
# 1) Basic operation when doing mechanistic interpretability: 
"""
Break open the black box model and look at all of the internal activations.
Done with ```logits, cache = model.run_with_cache(text)```

What does ```.run_with_cache``` actually do?
- Run the forward pass of the model, but also store all the internal activations in a cache.
"""
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = model.to_tokens(gpt2_text)
print(f"Tokens: {gpt2_tokens} | Device: {gpt2_tokens.device}")
gpt2_logits, gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)
"""
    * gpt2_text -- Logits: torch.Size([1, 33, 50257])
    * gpt2_tokens -- Logits: torch.Size([1, 33, 50257])
"""
    # ```remove_batch_dim=True```
    # Every activation inside the model begins with a batch dimension. 
    # We only entered a single batch dimension, that dimension is always 1, so passing in the argument ```remove_batch_dim=True``` removes that dimension from all activations in the cache.
    # ```gpt2_cache_no_batch_dim = gpt2_cache.remove_batch_dim()``` would do the same thing.
print(f"Logits: {gpt2_logits.shape} | Cache Keys: {list(gpt2_cache.keys())}")

# 2) Visualize the attention patterns of all the heads in layer 0, using Alan Cooney's CircuitsVis library.
"""
Things to Look at:
- The attention pattern in gpt2_cache, an ActivationCache object. (By entering in the name of the activation, followed by the layer index)
  Shape: [head_index, destination_position, source_position]
- By using ```models.to_str_tokens```, we can convert the text to a list of the string tokens, since there is an attention weight between each pair of tokens.
  -> Each pair of tokens? (i.e. the attention weight between the tokens at position i and j)
""" 
print(type(gpt2_cache)) # <class 'transformer_lens.ActivationCache.ActivationCache'>
# breakpoint()
attention_pattern = gpt2_cache["pattern", 0, "attn"]
# breakpoint()
"""
    How does the attention pattern look like?
        -> Attention pattern is a 3D tensor of shape [head_index, destination_position, source_position]
        
    
"""
print(f"Attention Patterns Shape: {attention_pattern.shape}")
gpt2_str_tokens = model.to_str_tokens(gpt2_text) # Convert the text to a list of string tokens
print(f"String Tokens: {gpt2_str_tokens}")

print("Layer 0 Head Attention Patterns:")
cv.attention.attention_pattern(tokens=gpt2_str_tokens, attention=attention_pattern)
# -> In this case, we only look at the layer 0 attention patterns, but storing the internal activations from all locations in the model.
# But storing all the internal activations from all locations in the model is expensive, so only collect the layer 0 attention patterns and stop the forward pass at the layer 1.
attn_hook_name = "blocks.0.attn.hook_pattern"
attn_layer = 0
_, gpt2_attn_cache = model.run_with_cache(gpt2_text, remove_batch_dim=True, stop_at_layer=attn_layer+1, names_filter=[attn_hook_name])
gpt2_attn = gpt2_attn_cache[attn_hook_name]
print(f"Attention Patterns Shape: {gpt2_attn.shape}") # Attention Patterns Shape: torch.Size([12, 33, 33])
assert torch.equal(gpt2_attn, attention_pattern)

# 3) Hooks: Interventing on Activations
"""
By interpreting neural networks mechanistically, we can exactly know exactly what operations are goining on inside.
-> 'Hook Points' is the key point.
Every activation inside the transformer is surrounded by a hook point, which allows us to edit or intervene on it.

At this phase, by adding a hook function to that activation, model runs as usual, but the hook function is applied to compute a replacement, and that is subsituted in for the activation.
Returns a tensor of the correct shape. 

*Note* Difference between PyTorch hooks and TransformerLens hooks:
- PyTorch hooks: Also act on a layer, and edit the input or output of that layer, or the gradient when applying autodiff.
- TransformerLens hooks: Same done as PyTorch hooks, but they are applied to every activations not layers.
"""
layer_to_ablate = 0
head_index_to_ablate = 8

# Define a head ablation hook
def head_ablation_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"], 
    hook: HookPoint
    )->Float[torch.Tensor, "batch pos head_index d_head"]:
    print(f"Applying head ablation hook to {hook.name} | Shape: {value.shape}")
    """
    Applying head ablation hook to blocks.0.attn.hook_v | Shape: torch.Size([1, 33, 12, 64])
    
    > Cache Keys: ['hook_embed', 'hook_pos_embed', 'blocks.0.hook_resid_pre', 
    >       'blocks.0.ln1.hook_scale', 'blocks.0.ln1.hook_normalized', 'blocks.0.attn.hook_q', 
    >       'blocks.0.attn.hook_k', 'blocks.0.attn.hook_v', 'blocks.0.attn.hook_attn_scores', ...
    blocks.0.attn.hook_v is the 8th head of the value vector in the attention mechanism of layer 0.
    """
    value[:, :, head_index_to_ablate, :] = 0.0 
    return value

original_loss = model(gpt2_text, return_type="loss")
ablated_loss = model.run_with_hooks(
    gpt2_tokens,
    return_type="loss",
    fwd_hooks=[(
        utils.get_act_name("v", layer_to_ablate),
        head_ablation_hook
    )]
) # ```run_with_hooks```: Run the model and temporarily add in the hook for just this run. 
print(f"Original Loss: {original_loss.item():.4f} | Ablated Loss: {ablated_loss.item():.4f}") # Original Loss: 3.9987 | Ablated Loss: 5.4528

clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = "After John and Mary went to the store, John gave a bottle of milk to"

clean_tokens = model.to_tokens(clean_prompt)
corrupted_tokens = model.to_tokens(corrupted_prompt)

def logits_to_logit_diff(logits, correct_answer=" John", incorrect_answer=" Mary"):
    correct_token_id = model.to_single_token(correct_answer)
    incorrect_token_id = model.to_single_token(incorrect_answer)
    logit_diff = logits[0, -1, correct_index] - logits[0, -1, incorrect_index]
    return logit_diff

# The author said that we run on the clean prompt with the cache so we store activations to patch in later.
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
clean_logit_diff = logits_to_logit_diff(clean_logits)
corrupted_logit_diff = logits_to_logit_diff(corrupted_logits)

print(f"Clean Logit Diff: {clean_logit_diff.item():.4f} | Corrupted Logit Diff: {corrupted_logit_diff.item():.4f}") # 

# 4) Patch in the Residual Stream
# We choose to act on the residual stream at the start of the layer, so we call it resid_pre
def residual_stream_patching_hook(
    resid_pre: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: int
) -> Float[torch.Tensor, "batch pos d_model"]:
    clean_resid_pre = clean_cache[hook.name]
    resid_pre[:, position, :] = clean_resid_pre[:, position, :]
    return resid_pre

# Make a tensor to store the results for each patching run. We put it on the model's device to avoid unnecessary transfers between CPU and GPU.
num_tokens = clean_tokens.shape[1] # len(clean_tokens[0])
ioi_patching_result = torch.zeros((model.cfg.n_layers, num_positions), device=model.cfg.device)

for layer in tqdm.tqdm(range(model.cfg.n_layers)):
    for position in range(num_positions):
        # Use functools.partial to create a temporary hook function with the position fixed
        temp_hook_fn = partial(residual_stream_patching_hook, position=position)
        # Run the model with the temporary hook function
        patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[
            (utils.get_act_name("resid_pre", layer), temp_hook_fn)
        ])
        # Calculate the logit difference for the patched model and store it in the results tensor
        patched_logit_diff = logits_to_logit_diff(patched_logits).detach()
        # Store the result, normalizing by the clean and corrupted logit differences so it's between 0 and 1
        ioi_patching_result[layer, position] = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
        
# Add the index to end of the label, because plotly doesn't like duplicate labels
token_labels = [f"{token}_{index}" for index, token in enumerate(model.to_str_tokens(clean_prompt))]
imshow(ioi_patching_result, x=token_labels, xaxis="Position", yaxis="Layer", title="Normalized Logit Difference After Patching Residual Stream on the IOI Task")

# 5) Accessing Activations
batch_size = 10
seq_len = 50
size = (batch_size, seq_len)
input_tensor = torch.randint(1000, 10000, size)

random_tokens = input_tensor.to(model.cfg.device)
repeated_tokens = einops.repeat(random_tokens, "batch seq_len -> batch (2 seq_len)")
repeated_logits = model(repeated_tokens)

corrected_log_probs = model.loss_fn(repeated_logits, repeated_tokens, per_token=True)
loss_by_position = einops.reduce(corrected_log_probs, "batch position -> position", "mean")
line(loss_by_position, xaxis="Position", yaxis="Loss", title="Loss by position on random repeated tokens")

# We make a tensor to store the induction score for each head. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
def induction_score_hook(
    pattern: Float[torch.Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    # We take the diagonal of attention paid from each destination position to source positions seq_len-1 tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    # Get an average score per head
    induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    # Store the result.
    induction_score_store[hook.layer(), :] = induction_score

# We make a boolean filter on activation names, that's true only on attention pattern names.
pattern_hook_names_filter = lambda name: name.endswith("pattern")

model.run_with_hooks(
    repeated_tokens, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

imshow(induction_score_store, xaxis="Head", yaxis="Layer", title="Induction Score by Head")

if IN_GITHUB:
    torch.manual_seed(50)
    
induction_head_layer = 5
induction_head_index = 5
size = (1, 20)
input_tensor = torch.randint(1000, 10000, size)

single_random_sequence = input_tensor.to(model.cfg.device)
repeated_random_sequence = einops.repeat(single_random_sequence, "batch seq_len -> batch (2 seq_len)")
def visualize_pattern_hook(
    pattern: Float[torch.Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    display(
        cv.attention.attention_patterns(
            tokens=model.to_str_tokens(repeated_random_sequence), 
            attention=pattern[0, induction_head_index, :, :][None, :, :] # Add a dummy axis, as CircuitsVis expects 3D patterns.
        )
    )

model.run_with_hooks(
    repeated_random_sequence, 
    return_type=None, 
    fwd_hooks=[(
        utils.get_act_name("pattern", induction_head_layer), 
        visualize_pattern_hook
    )]
)