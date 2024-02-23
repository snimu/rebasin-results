"""
Code for testing rebasin.
Initially taken from https://github.com/tysam-code/hlb-gpt

"""


# Note: The main change we need to make if we're in Colab is to uncomment this below block (and to add !pip install tiktoken to a cell above).
# If we are in an ipython session or a notebook, clear the state to avoid bugs

# To enable torch.compile in the code, don't forget to upgrade to torch 2.0! It's out of prerelease now so not too hard of an install command anymore:
#pip3 install --upgrade torch

"""
# don't forget these too:
# !pip3 install tiktoken
# If you don't have torch 2.0 on whatever environment you're using:
# !pip3 install --upgrade torch
try:
  _ = get_ipython().__class__.__name__
  ## we set -f below to avoid prompting the user before clearing the notebook state
  %reset -f
except NameError:
  pass ## we're still good
"""
import argparse
import copy
from functools import partial
import urllib
import zipfile
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
import rebasin
import pandas as pd

# This seems like one of the best choices right now for a fast/lightweight/simple tokenizer.
import tiktoken

# Check if we're using pytorch 2 for those speedups
using_pytorch_2 = (int(torch.__version__.split('.')[0]) >= 2)
if not using_pytorch_2:
    print("Info: Pytorch 2.0 isn't currently installed. Falling back to slower Pytorch 1.x pathway.")

## <-- teaching comments
# <-- functional comments
# You can run 'sed -i.bak '/\#\#/d' ./main.py' to remove the teaching comments if they are in the way of your work. <3

# This can go either way in terms of actually being helpful when it comes to execution speed.
# torch.backends.cudnn.benchmark = True

# This code was built from the ground up to be directly hackable and to support rapid experimentation, which is something you might see
# reflected in what would otherwise seem to be odd design decisions. It also means that maybe some cleaning up is required before moving
# to production if you're going to use this code as such (such as breaking different section into unique files, etc). Think of this project
# as a digital breadboard for your research. :D :) That said, if there's ways this code could be improved and cleaned up, please do open a
# PR on the GitHub repo. Your support and help is much appreciated for this project! :)

# This is for testing that certain changes don't exceed X% portion of the reference GPU (here an A100)
# so we can help reduce a possibility that future releases don't take away the accessibility of this codebase.
#torch.cuda.set_per_process_memory_fraction(fraction=TBD./40., device=0) ## 40. GB is the maximum memory of the base A100 GPU

## Useful for debugging gradient issues (nan values and such). Warning -- its slow!
# torch.autograd.set_detect_anomaly(True)

# If we run out of memory, we can always increase the accumulate_steps and decrease this by the same factor (2x, 4x, etc).
# Currently, doing this doesn't result in _exact_ equivalence due to a quirk of implementation, but it should at some point in the future.
# NOTE: The more batchsize we can use, the better. Assumes this batchsize is the target value at hyp['misc']['sequence_length']['max'].
batchsize = 64

# The default model here below is roughly ~30.82M parameters or so.
hyp = {
    'opt': {
        'lr': 2e-3,
        'weight_decay': 3e-2,
        'total_train_steps': 2_000,
        'eval_iter': 50, # how many train iterations we wait in between eval rounds (we don't include eval time in our performance stats)
        'warmup_percent': .001, ## what percent of the training run to warmup the learning rate over
        'initial_accumulate_steps': 1, # It's good for speed to start small (i.e. 1) here and tune target_per_step_decay carefully to grow to the appropriate num of accumulate steps over traiing
    },
    'net': {
        'residual_depth': 384, ## this should be a factor of 8 in some way to stay tensor core friendly
        'num_heads': 3,
        'num_blocks': 6,
    },
    'misc': {
        'num_tokens': 50304, # Rounded to the nearest value of 64 for dose sheeeeer speeeeeddssss :D
        'sequence_length': {
            'max': 512,
            'initial': 32, # Very short initial sequence length,
            'growth_steps': 195, # Increase the sequence length every n steps, up to the maximum limit
        },
        'device': 'cuda',
        'dtype': torch.bfloat16,
        'data_location': 'data.pt',
    }
}

#############################################
#                Dataloader                 #
#############################################

if not os.path.exists(hyp['misc']['data_location']):
    print("downloading data and tokenizing (1-2 min)")

    raw_data_source = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip'
    raw_data_cache = './data_raw/' # where to cache the data after downloading

    if not os.path.isfile(raw_data_cache):
        os.makedirs(raw_data_cache, exist_ok=True)
        urllib.request.urlretrieve(raw_data_source, raw_data_cache+'data.zip')

    with zipfile.ZipFile('data_raw/data.zip', 'r') as zip_ref:
        zip_ref.extractall('data_raw/')

    with open('data_raw/wikitext-103-raw/wiki.train.raw', 'r', encoding="utf8") as data_file:
        raw_train_data = data_file.read()

    with open('data_raw/wikitext-103-raw/wiki.valid.raw', 'r', encoding="utf8") as data_file:
        raw_eval_data = data_file.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    raw_tokenized_train = tokenizer.encode_ordinary(raw_train_data)
    raw_tokenized_eval = tokenizer.encode_ordinary(raw_eval_data)

    train_tokenized = torch.tensor(raw_tokenized_train, device=hyp['misc']['device'], dtype=torch.int) # int64 is likely overkill for the amount of tokens we have...
    eval_tokenized = torch.tensor(raw_tokenized_eval, device=hyp['misc']['device'], dtype=torch.int)

    data = {
        'train': train_tokenized,
        'eval': eval_tokenized
        }

    torch.save(data, hyp['misc']['data_location'])
    print("completed the tokenization process!")

else:
    ## This is effectively instantaneous, and takes us practically straight to where the dataloader-loaded dataset would be. :)
    ## So as long as you run the above loading process once, and keep the file on the disc it's specified by default in the above
    ## hyp dictionary, then we should be good. :)
    data = torch.load(hyp['misc']['data_location'])


## As you'll note above and below, one difference is that we don't time loading the raw data to GPU since it's such a variable operation
## (this includes the tokenizing), and it can sort of get in the way of measuring other things.
#############################################
#            Network Components             #
#############################################

class LayerNorm(nn.Module):
    """ A variant of LayerNorm that lets us turn our weights and bias trainable values on and off with true and false, respectively"""
    def __init__(self, num_features, eps=1e-5, weight=True, bias=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features)) if weight else None
        self.bias = nn.Parameter(torch.zeros(num_features)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, weight=self.weight, bias=self.bias, eps=self.eps)

class AttentionBlock(nn.Module):
    """ A standard attention block (for now....?) """
    def __init__(self, num_features, sequence_length, num_heads):
        super().__init__()
        self.norm = LayerNorm(num_features, bias=False)
        # this function below is complicated and can be very tricky in instantiation and in calling, as it is implemented
        # strangely and the parameters are relatively unintuitive in how they are passed. before making any changes,
        # be sure to read https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html thoroughly
        self.attention = nn.MultiheadAttention(num_features, num_heads, bias=False, batch_first=True)

        # Below we set up learnable linear encodings. Similar to https://arxiv.org/abs/2108.12409
        self.linear_encoding_lr_mult = 100. # hardcoded for now
        self.linear_encoding_scaler = nn.Parameter(torch.tensor(1./self.linear_encoding_lr_mult, device='cuda'))
        # Note: This is expensive to store for each layer but should be okay for now. #TODO is to refactor if memory becomes an issue for us
        self.linear_encoding_base = (torch.arange(-sequence_length+1, 1, dtype=torch.bfloat16, device=hyp['misc']['device'])).unsqueeze(0) + torch.arange(sequence_length-1, -1, step=-1, dtype=torch.bfloat16, device=hyp['misc']['device']).unsqueeze(1)
        self.linear_encoding_mask = lambda mask, encoding_base, scaler: torch.where(mask, F.softplus(self.linear_encoding_lr_mult*scaler)*encoding_base, torch.empty_like(encoding_base, dtype=torch.bfloat16).fill_(-float("inf")))
        ## this mask makes sure that each part of a sequence can only attend to the tokens that come behind it.
        self.causal_mask = torch.tril(torch.ones((sequence_length, sequence_length), device=hyp['misc']['device'], dtype=torch.bool)) #torch.logical_not(torch.triu(torch.ones((sequence_length, sequence_length), device=hyp['misc']['device'], dtype=torch.bool))).T # TODO: way to simplify this? (see: after pytorch 2.0 release, causal=True on the scaled_dot_product_attention fn)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        attn_mask = self.linear_encoding_mask(self.causal_mask, self.linear_encoding_base, self.linear_encoding_scaler)
        ## https://youtu.be/kCc8FmEb1nY?t=3720 is a good explanation of self-attention
        x, _ = self.attention(x, x, x, attn_mask=attn_mask[:x.shape[1], :x.shape[1]], need_weights=False)
        x = x + residual # haiku
        return x

class SiGLU(nn.Module):
    """ Implements the SiLU-gated linear unit from that one gated linear units paper. Assumes the channel tensors are stacked. """
    def __init__(self):
        super().__init__()
        self.activation = nn.SiLU()

    def forward(self, x, dim=-1):
        x = x.split((x.shape[-1]//2, x.shape[-1]//2), dim=dim)
        x = x[0] * self.activation(x[1])
        return x

class MLPBlock(nn.Module):
    """ A standard MLP block (for now....?) """
    def __init__(self, num_channels, expansion_factor=3):
        super().__init__()
        self.norm = LayerNorm(num_channels, bias=False)
        self.expand = nn.Linear(num_channels, num_channels*2*expansion_factor, bias=False)
        self.project = nn.Linear(expansion_factor*num_channels, num_channels, bias=False)
        self.siglu = SiGLU()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.expand(x)
        x = self.siglu(x)
        x = self.project(x)
        x = x + residual # haiku
        return x


#############################################
#          Init Helper Functions            #
#############################################

# Nothing here but us chickens! :D

#############################################
#            Network Definition             #
#############################################

class SpeedyLangNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict # flexible, defined in the make_net function

    # This allows you to customize/change the execution order of the network as needed.
    def forward(self, x):
        x = self.net_dict['embedding'](x) # Look up the input embeddings
        x = self.net_dict['connection'](x)  # Linear layer s.t. there are permutations!
        for block in range(hyp['net']['num_blocks']):
            x = self.net_dict['attn_layers'][block](x) # note: residuals are included in the block definitions for these layers
            x = self.net_dict['mlp_layers'][block](x)  # note: residuals are included in the block definitions for these layers
        x = self.net_dict['norm'](x)
        x = self.net_dict['outputs'](x)
        return x

def make_net():
    # Note, you have to specify any arguments overlapping with defaults (i.e. everything but in/out depths) as kwargs so that they are properly overridden (TODO cleanup somehow?)
    network_dict = nn.ModuleDict({
        'embedding': nn.Embedding(hyp['misc']['num_tokens'], hyp['net']['residual_depth'], scale_grad_by_freq=True),
        'connection': nn.Linear(hyp['net']['residual_depth'], hyp['net']['residual_depth'], bias=False),
        'norm': LayerNorm(hyp['net']['residual_depth'], bias=False),
        'mlp_layers': nn.ModuleList([MLPBlock(hyp['net']['residual_depth']) for _ in range(hyp['net']['num_blocks'])]),
        'attn_layers': nn.ModuleList([AttentionBlock(hyp['net']['residual_depth'], hyp['misc']['sequence_length']['max'], hyp['net']['num_heads']) for _ in range(hyp['net']['num_blocks'])]),
        'outputs': nn.Linear(hyp['net']['residual_depth'], hyp['misc']['num_tokens'], bias=False),
    })

    net = SpeedyLangNet(network_dict)
    net = net.to(hyp['misc']['device'], torch.bfloat16)
    net.train()

    # Tie the input and output weights. Feel free to experiment with this!
    net.net_dict['embedding'].weight = net.net_dict['outputs'].weight

    for name, parameter in net.named_parameters():
        # TODO: Way to tidy this up for a future release? (once pytorch 2.0 releases we can use the scaled_dot_product attention, update the names appropriately, and point to an older release for people using PT <2.0)
        # Initialize both embedding layers (embedding and position) and the non-bias values of the 'normal' linear layers (outputs, expand, in_proj)
        if 'embedding' in name or 'position' in name or (('expand' in name or 'in_proj' in name or 'outputs' in name) and 'weight' in name):
            torch.nn.init.normal_(parameter.data, mean=0., std=.02) # normal init
        elif ((('project' in name and 'mlp' in name) or 'out_proj' in name) and 'weight' in name):
            # As noted in NanoGPT, this is from the GPT-2 paper. Also very similar from what I seee to the FixUp initialization for ResNets
            torch.nn.init.normal_(parameter.data, mean=0., std=.02/((2 * hyp['net']['num_blocks'])**.5)) # keeps variance from exploding when adding to the residual
        elif 'norm' in name or 'scaler' in name:
            pass # the norms already get initialized to values that we want -- we just include this so that we can warn the user if they add a new param
                 # that isn't initialized correctly in the future here.
        elif 'connection' in name:
            pass
        else:
            print(f"warning, no initialization keyword match for: {name}!")

    # We compile the network later so that we can include compilation time inside of the training time to be an honest comparison against other methods.
    return net


def get_net_mfu_and_param_counts(net, current_batchsize, current_sequence_length, gradient_accumulation_steps, avg_time_per_batch):
    assert hyp['misc']['dtype'] in (torch.half, torch.bfloat16), "Flops calculation is inaccurate with types other than half-precision types"
    flops_dict = {}
    # The below is a very easy way to estimate total registered param counts. I believe the reason that we don't count the embedding-only layers by default
    # is because they function as a lookup table (so, technically not really FLOPs at all)
    params_dict = {name: parameter.numel() if not 'position' in name else 0 for name, parameter in net.named_parameters()}
    total_num_params = sum(params_dict.values())

    # Rough flops estimate, see https://github.com/karpathy/nanoGPT/blob/ae3a8d5fdd3ddb8b13fab182723476523961e3ab/model.py#L327 for more info
    # Originally sourced from the PaLM paper, appendix B: https://arxiv.org/abs/2204.02311
    # This includes both the forward and backwards passes :D
    flops_for_single_input_token = 6 * total_num_params + 12 * hyp['net']['num_blocks'] * hyp['net']['residual_depth'] * current_sequence_length
    flops_for_full_sequence = flops_for_single_input_token * current_sequence_length
    flops_for_single_step = flops_for_full_sequence * current_batchsize * gradient_accumulation_steps

    current_flops_achieved = flops_for_single_step/avg_time_per_batch
    a100_total_possible_flops = 312e12 # TODO: is there a good way to make this more flexible?
    mfu_value = current_flops_achieved / a100_total_possible_flops

    return mfu_value, params_dict, total_num_params


#############################################
#            Data Preprocessing             #
#############################################

# Nothing here for the baseline...but us chickens! :D

########################################
#          Training Helpers            #
########################################
@torch.no_grad()
def get_batches(data_dict, key, batchsize, sequence_length, num_steps):
    # Generator version of get_batch which is really nice for a for loop if we have static arguments
    # So, ~!NOTE!~ This means that we assume a fixed batchsize and sequence size for this function!
    # We recommend using the (unfortunately slightly less fancy) get_batch function for other usecases.
    total_steps_in_iter = min(len(data_dict[key])//batchsize, num_steps) # the complete number of batches to do
    for _ in range(total_steps_in_iter):
        yield get_batch(data_dict, key, batchsize, sequence_length)


# Get a single batch item. Currently used in the training loop
def get_batch(data_dict, key, batchsize, sequence_length):
    shuffled = torch.randint(len(data_dict[key]) - sequence_length - 1, (batchsize,), device=hyp['misc']['device'])
    batch_index_offsets = torch.arange(0, sequence_length, dtype=torch.long, device=hyp['misc']['device']) # Create offsets so we can index every item in the sequence
    batch_indexes = shuffled.unsqueeze(-1) + batch_index_offsets.unsqueeze(0)

    # Unfortunately for now we have to flatten this since this seems to be the best way to sample it (then we have to reshape it again at the end)
    batch_indexes_flattened = batch_indexes.flatten()

    return torch.take_along_dim(data_dict[key], batch_indexes_flattened,   dim=0).view(batchsize, sequence_length).long(), \
           torch.take_along_dim(data_dict[key], batch_indexes_flattened+1, dim=0).view(batchsize, sequence_length).long() # Returns each token as the input, then the _next_ token in the sequence as a target


def init_split_parameter_dictionaries(net):
    params_non_decay = {'params': [], 'lr': hyp['opt']['lr'], 'eps': 1e-9, 'betas': (.9, .95), 'weight_decay': 0.}
    params_decay     = {'params': [], 'lr': hyp['opt']['lr'], 'eps': 1e-9, 'betas': (.9, .95), 'weight_decay': hyp['opt']['weight_decay']}

    for name, p in net.named_parameters():
        if p.requires_grad:
            if 'outputs' in name or 'norm' in name or 'embedding' in name or 'bias' in name:
                params_non_decay['params'].append(p)
            else:
                params_decay['params'].append(p) # Default to weight decay unless we explicitly specify that we don't want to use it

    return params_non_decay, params_decay


def get_grad_norm(net):
    # Gets the entire grad norm of the network.
    grad_norm = torch.tensor(0., device=hyp['misc']['device'])
    for p in net.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.square()
    grad_norm = (grad_norm ** 0.5).item()
    return grad_norm


def grow_sequence_length(current_sequence_length, current_max_batchsize, current_batchsize):
    old_sequence_length = current_sequence_length
    # Dynamically grows the sequence length and updates the relevant parameters during training
    current_sequence_length *= 2
    current_sequence_length = min(current_sequence_length, hyp['misc']['sequence_length']['max'])
    current_max_batchsize = round(batchsize * hyp['misc']['sequence_length']['max']/current_sequence_length)

    old_batchsize = current_batchsize
    if current_batchsize >= current_max_batchsize: # if we're at our peak and we're doubling our sequence length, half the batchsize to keep from OOMing, and then update the virtual steps accordingly
        current_batchsize = min(current_batchsize // 2, current_max_batchsize)

    print(f"| increasing sequence length (old: {old_sequence_length}, new: {current_sequence_length}), adjusting batchsize as necessary to fit (old: {old_batchsize}, new: {current_batchsize}, current_maximum: {current_max_batchsize})")

    return current_sequence_length, current_max_batchsize, current_batchsize

## Just your good ol', normal an' nice xentropy function. Which makes sense if (in the ideal scenario) we only see each datapoint one single time.
## However! If (esp for tiny datsets) we're seeing our data multiple times in a row, then maybe some smoothing to help regularize things a bit is in order.... :D
loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)

logging_columns_list = ['epoch', 'current_steps', 'train_loss', 'val_loss', 'val_perplexity', 'train_acc', 'val_acc', 'grad_norm', 'a100_mfu', 'total_time_seconds']
# define the printing function and print the column heads
def print_training_details(columns_list, separator_left='|  ', separator_right='  ', final="|", column_heads_only=False, is_final_entry=False):
    print_string = ""
    if column_heads_only:
        for column_head_name in columns_list:
            print_string += separator_left + column_head_name + separator_right
        print_string += final
        print('-'*(len(print_string))) # print the top bar
        print(print_string)
        print('-'*(len(print_string))) # print the bottom bar
    else:
        for column_value in columns_list:
            print_string += separator_left + column_value + separator_right
        print_string += final
        print(print_string)
    if is_final_entry:
        print('-'*(len(print_string))) # print the final output bar

print_training_details(logging_columns_list, column_heads_only=True) ## print out the training column heads before we print the actual content for each run.

# We basically need to look up local variables by name so we can have the names, so we can pad to the proper column width.
## Printing stuff in the terminal can get tricky and in a previous life, this used an outside library. But some of the required stuff
## for that seemed even more heinous than this, unfortunately. So we switched to the "more simple" version of this!
format_for_table = lambda x, locals: (f"{locals[x]}".rjust(len(x))) \
                                          if x in locals and type(locals[x]) == int else "{:0.4f}".format(locals[x]).rjust(len(x)) \
                                      if x in locals and locals[x] is not None \
                                      else " "*len(x)

########################################
#           Train and Eval             #
########################################

def eval(net):
    ####################
    # Evaluation  Mode #
    ####################

    eval_batchsize = 64 # Note/semi-warning: This is slightly 'buggy' technically as it will drop the last batch in the eval set, but that should just add a bit of noise for our usecases
    num_steps = 32 # Do a slightly noisy fast eval (that should be good enough for our purposes)
    loss_list_val, acc_list = [], []

    with torch.no_grad():
        # Note: We eval at the maximum sequence length so that we can get an idea of how well the sequence length growing scales (generally pretty well, for here at least! :D)
        for inputs, targets in get_batches(data, key='eval', batchsize=eval_batchsize, sequence_length=hyp['misc']['sequence_length']['max'], num_steps=num_steps):
            outputs = net(inputs)
            val_loss = loss_fn(outputs.flatten(0, 1), targets.flatten(0, 1))
            loss_list_val.append(val_loss)
            acc_list.append((outputs.argmax(-1) == targets).float().mean())

        val_acc = torch.stack(acc_list).mean().item()
        val_loss = torch.stack(loss_list_val).mean().item()
        val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity


def eval_full(net):
    ####################
    # Evaluation  Mode #
    ####################

    eval_batchsize = 64 # Note/semi-warning: This is slightly 'buggy' technically as it will drop the last batch in the eval set, but that should just add a bit of noise for our usecases
    num_steps = 1e12 # Use all validation data
    loss_list_val, acc_list = [], []

    with torch.no_grad():
        # Note: We eval at the maximum sequence length so that we can get an idea of how well the sequence length growing scales (generally pretty well, for here at least! :D)
        for inputs, targets in get_batches(data, key='eval', batchsize=eval_batchsize, sequence_length=hyp['misc']['sequence_length']['max'], num_steps=num_steps):
            outputs = net(inputs)
            val_loss = loss_fn(outputs.flatten(0, 1), targets.flatten(0, 1))
            loss_list_val.append(val_loss)
            acc_list.append((outputs.argmax(-1) == targets).float().mean())

        val_acc = torch.stack(acc_list).mean().item()
        val_loss = torch.stack(loss_list_val).mean().item()
        val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity


def perplexity(net, device):
    val_acc, val_loss, val_perplexity = eval(net)
    return val_perplexity


def model_loss(net, device):
    val_acc, val_loss, val_perplexity = eval(net)
    return val_loss


def accuracy(net, device):
    val_acc, val_loss, val_perplexity = eval(net)
    return val_acc


def perplexity_full(net, device):
    val_acc, val_loss, val_perplexity = eval_full(net)
    return val_perplexity


def model_loss_full(net, device):
    val_acc, val_loss, val_perplexity = eval_full(net)
    return val_loss


def accuracy_full(net, device):
    val_acc, val_loss, val_perplexity = eval_full(net)
    return val_acc


def train_model():
    # Initializing variables for the whole run.
    total_time_seconds = 0.
    microbatch_step = current_steps = 0
    microbatches_since_last_eval = 0. # TODO: Good way to simplify this?
    tokens_seen = 0

    # Dynamic growth-related parameters
    grad_norm = previous_grad_norm = 2. # initialize the grad norm calculation to roughly the initial grad norm
    target_per_step_decay = 3.6e-2 # what absolute step size we should target each training step. the effective batchsize is scaled to try to meet this target. :)
    accumulate_steps_lr = 1e-1 # smooths out the automatic batchsize scaling rate
    current_accumulate_steps = accumulate_steps_estimate = hyp['opt']['initial_accumulate_steps'] # current_accumulate_steps is the per-microbatch sampled steps, accumulate_steps_estimate is the actual estimated fractional value determining our projected batchsize
    current_sequence_length = hyp['misc']['sequence_length']['initial']
    # Start at the maximum allowable batchsize, which is the base batchsize (assumes max sequence length) times the ratio of the max sequence length to the shortest sequence length
    current_batchsize = batchsize * round(hyp['misc']['sequence_length']['max']/current_sequence_length)

    # Note: This is a static calculation of the total number of microbatches up front, you may have to change this depending upon what you're tinkering with
    total_microbatch_steps = hyp['opt']['total_train_steps'] * hyp['opt']['initial_accumulate_steps'] # BUG: Since we have dynamic virtual batchsize scaling now, we're going to have to rewrite the dataloader to appropriately handle it now.

    # Get network
    net = make_net()

    ## Stowing the creation of these into a helper function to make things a bit more readable....
    params_non_decay, params_decay = init_split_parameter_dictionaries(net)
    adamw_speedup_mode = {'fused': True} if using_pytorch_2 else {'foreach': True}
    opt = torch.optim.AdamW([params_non_decay, params_decay], **adamw_speedup_mode)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=hyp['opt']['lr'], total_steps=hyp['opt']['total_train_steps'], pct_start=hyp['opt']['warmup_percent'], anneal_strategy='linear', cycle_momentum=False, div_factor=1e2, final_div_factor=.02)

    ## For accurately timing GPU code
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize() ## clean up any pre-net setup operations
    starter.record()

    # If you have pytorch 2.0, compiling the network before training can help us a ton
    # If you need to/are brave enough, check the top of this file for a pip command to install it.
    # It can bork your pre-existing setup though if you're not careful, so bewarb! D: :D

    # Since we're dynamically changing the sequence length during training, we've turned off the compiling since that's faster for now.
    if False: #using_pytorch_2:
        net = torch.compile(net)

    #################
    # Training Mode #
    #################
    net.train()

    val_loss = None
    val_acc = None
    val_perplexity = None

    batch_iter_kwargs = {'data_dict': data, 'key': 'train', 'batchsize': batchsize, 'num_steps': total_microbatch_steps, 'sequence_length': hyp['misc']['sequence_length']}

    # Step nearly infinitely, as our breaking condition is inside the loop now
    while current_steps < hyp['opt']['total_train_steps']:
        # Limit the batchsize each step to keep GPU memory from exploding (TODO might be to consolidate this into the 'grow_sequence_length' function if that ends up being the only place that this variable is primarily relevant)
        current_max_batchsize = round(batchsize * hyp['misc']['sequence_length']['max']/current_sequence_length)
        inputs, targets = get_batch(data, key='train', batchsize=current_batchsize, sequence_length=current_sequence_length)

        outputs = net(inputs)

        loss = loss_fn(outputs.flatten(0, 1), targets.flatten(0, 1))

        # Quick non-eval summary every N training steps
        if current_steps % 10 == 0 and microbatch_step % current_accumulate_steps == 0 and not current_steps % hyp['opt']['eval_iter'] == 0:
            train_acc = (outputs.detach().argmax(-1) == targets).float().mean().item()
            train_loss = loss.detach().cpu().item()
            train_summary_variables = {'epoch': tokens_seen//len(data['train']), 'current_steps': current_steps, 'train_loss': train_loss, 'train_acc': train_acc, 'grad_norm': grad_norm}
            print_training_details(list(map(partial(format_for_table, locals=train_summary_variables), logging_columns_list)))

        loss.div(current_accumulate_steps).backward()
        tokens_seen += current_batchsize * current_sequence_length
        microbatches_since_last_eval += 1

        ## Once we've accumulated steps over all of our microbatches, take a single full-batchsize step.
        if microbatch_step % current_accumulate_steps == 0:
            ## Step the optimizer, then scheduler
            opt.step()
            # Dynamic weight decay scheduling. Based upon the inverse perplexity of the network over the data [inspired by section 5 of https://arxiv.org/pdf/2204.02311.pdf]
            # (up to its max value at perplexity = 0, which we should in all...likelihood...never reach. :')))) )
            # Still evaluating the top-end of this option vs a few other options out there.
            opt.param_groups[1]['weight_decay'] = (1./loss.detach().item())**2. * hyp['opt']['weight_decay']
            scheduler.step()

            # Check to see if we need to grow our batchsize according to the scheduler (or some potential future growing criteria :D)
            if current_steps % hyp['misc']['sequence_length']['growth_steps'] == 0 and current_steps != 0 and current_sequence_length < hyp['misc']['sequence_length']['max']:
                current_sequence_length, current_max_batchsize, current_batchsize = grow_sequence_length(current_sequence_length, current_max_batchsize, current_batchsize)

            # The next several lines calculate a dynamic batchsize, simulated through manual dithering
            # There could be improvements or losses in changing the dithering strategy, since determinism and gradient descent can lead to some very not-so-nice (and subtle) loss oscillations.
            # First, manually calculate the grad norm here (no clipping or anything)
            grad_norm = get_grad_norm(net) # TODO: Can/should we evaluate every N steps instead?

            per_step_diff_delta = target_per_step_decay - (previous_grad_norm - grad_norm)
            previous_grad_norm = grad_norm
            # Scale the learning rate by the current number of accumulate steps so we're able to be nimble even if steps take a very long time
            accumulate_steps_estimate += current_accumulate_steps * (accumulate_steps_lr * per_step_diff_delta)
            # Clamp our fractional accumulate steps estimate so it doesn't go below 1
            accumulate_steps_estimate = max(1., accumulate_steps_estimate)
            base, probability = divmod(accumulate_steps_estimate, 1)
            # Randomly sample next accumulate steps to use
            current_accumulate_steps = max(1, int(base + torch.bernoulli(torch.tensor(probability)).item())) # bernoulli via torch to save an unnecesary import :)

            ## Using 'set_to_none' I believe is slightly faster (albeit riskier w/ funky gradient update workflows) than under the default 'set to zero' method
            opt.zero_grad(set_to_none=True)
            current_steps += 1

            # Since we're not running over epochs anymore, we have to manually calculate what epoch it is.
            epoch = tokens_seen//len(data['train'])

            if current_steps % hyp['opt']['eval_iter'] == 0:
                ender.record()
                torch.cuda.synchronize()
                total_time_seconds += 1e-3 * starter.elapsed_time(ender)
                train_loss = loss.detach().cpu().item() # To have an updated loss to compare with the eval loss

                opt.zero_grad(set_to_none=True)
                # Potential # bug warning: We're disabling the eval switch here as the nn.MultiheadAttention class fails in eval mode w/ a pure bfloat16 network. Functionally they should be the same; however, care should be taken if implementing something with clear differences between train and eval, like dropout.
                #net.eval()

                val_acc, val_loss, val_perplexity = eval(net)
                average_time_per_batch = 1e-3 * starter.elapsed_time(ender)/hyp['opt']['eval_iter']
                # You can use this variable to print out the parameter counts of the network if you want, though we aren't printing this out in this particular version.
                a100_mfu, _, param_counts = get_net_mfu_and_param_counts(net, current_batchsize, current_sequence_length, microbatches_since_last_eval/hyp['opt']['eval_iter'], avg_time_per_batch=average_time_per_batch)
                microbatches_since_last_eval = 0 # necessary for accurate mfu counts. How totally necessary is mfu here if we're mainly using wallclock time?
                is_final_eval = (current_steps == hyp['opt']['total_train_steps']) # If we're at the end of training, do a full eval instead

                # Print out our training details (sorry for the complexity, the whole logging business here is a bit of a hot mess once the columns need to be aligned and such....)
                ## We also check to see if we're on our final eval loop (assuming that max_num_steps lines up with the eval_iter value) so we can print the 'bottom' of the table for each round.
                print_training_details(list(map(partial(format_for_table, locals=locals()), logging_columns_list)), is_final_entry=is_final_eval)
                torch.cuda.synchronize()
                starter.record()
                net.train() # Functionally shouldn't do anything with the base network, just adding this to guard against any bugs for any future changes that do require this <3 <3 <3
        microbatch_step += 1

    return net, val_loss # Return the final validation loss achieved (not using the 'best validation loss' selection strategy, which I think is okay here....)


def test_metrics(model):
    acc, loss, perp = eval_full(model)
    acc_ = accuracy_full(model, None)
    loss_ = model_loss_full(model, None)
    perp_ = perplexity_full(model, None)

    print(f"Accuracy: {acc} vs {acc_}")
    print(f"Loss: {loss} vs {loss_}")
    print(f"Perplexity: {perp} vs {perp_}")
    print(f"{abs((acc - acc_) / acc)=}")

    # Only small percentage differences are allowed
    assert -0.1 < (acc - acc_) / acc < 0.1
    assert -0.1 < (loss - loss_) / loss < 0.1
    assert -0.1 < (perp - perp_) / perp < 0.1

    return acc, loss, perp


def test_memory_capacity():
    """Test that three models can fit into memory --- important for interpolation!"""
    model_a = make_net()
    model_b = make_net()
    model_c = make_net()

    model_a = model_a.to("cuda" if torch.cuda.is_available() else "cpu")
    model_b = model_b.to("cuda" if torch.cuda.is_available() else "cpu")
    model_c = model_c.to("cuda" if torch.cuda.is_available() else "cpu")

    # And one forward pass per model
    input_data, _ = get_batch(
        data, key='eval', batchsize=64, sequence_length=32
    )
    model_a(input_data)
    model_b(input_data)
    model_c(input_data)

    del model_a
    del model_b
    del model_c


def test_rebasin_works():
    ma = make_net()
    mb = make_net()
    mbo = copy.deepcopy(mb).to("cuda:0")
    ma, mb = ma.to("cuda:0"), mb.to("cuda:0")
    input_data, _ = get_batch(
        data, key='eval', batchsize=64, sequence_length=32
    )

    pcd = rebasin.PermutationCoordinateDescent(
        ma, mb, input_data, logging_level="info", device_b="cuda:0"
    )

    pcd.rebasin()

    # Make sure that calling the models twice gives the same result both times
    assert torch.allclose(mb(input_data), mb(input_data))

    print((mb(input_data) - mbo(input_data)).abs().sum() / mbo(input_data).abs().sum())

    # Make sure that the permuted model produces the same output as the original model
    assert torch.allclose(mb(input_data), mbo(input_data), atol=1e-5, rtol=1e-4)

    # Make sure that the permuted model actually changed
    sum_diff_perc = 0
    num_params = 0
    for p1, p2 in zip(mb.parameters(), mbo.parameters()):
        sum_diff_perc += (p1 -p2).abs().sum().item() / p2.abs().sum().item()
        num_params += 1

    sum_diff_perc /= num_params
    assert sum_diff_perc > 0.1

    # Cleanup
    del mbo

    # Make sure that interpolation at least doesn't throw an exception
    interp = rebasin.interpolation.LerpSimple(
        models=[ma, mb],
        devices=["cuda:0", "cuda:0"],
        logging_level="info",
        device_interp="cuda:0",
    )
    interp.interpolate(steps=1)

    # Cleanup
    del ma, mb


def eval_saved_models(directory: str):
    all_filenames = next(os.walk(directory))[2]

    losses = []
    perplexities = []
    accuracies = []

    model = make_net()
    loop = tqdm(all_filenames)
    for filename in loop:
        loop.set_description(filename)
        model.load_state_dict(torch.load(os.path.join(directory, filename)))
        a_, l_, p_ = eval_full(model)
        losses.append(l_)
        perplexities.append(p_)
        accuracies.append(a_)

    return losses, perplexities, accuracies


def test_rebasin(interp_steps: int = 99):
    # # Test metrics - fail early if there is a problem
    # # Don't run now, seems to work and takes long
    # print("Testing metrics")
    # model = make_net()
    # test_metrics(model)
    # del model  # Free memory

    # print("Test that rebasin works.")
    # test_rebasin_works()
    # print("Done.")

    print("Testing memory capacity")
    test_memory_capacity()
    print("Done.\nTesting rebasin")

    # Train
    model_a, _ = train_model()
    acc_a, loss_a, perp_a = eval_full(model_a)
    model_b, _ = train_model()
    model_b_original = copy.deepcopy(model_b)
    acc_bo, loss_bo, perp_bo = eval_full(model_b_original)

    # Rebasin
    input_data, _ = get_batch(
        data, key='eval', batchsize=64, sequence_length=32
    )

    pcd = rebasin.PermutationCoordinateDescent(
        model_a,
        model_b,
        input_data,
        logging_level="info",
        device_b="cuda:0",
    )
    pcd.calculate_permutations()
    pcd.apply_permutations()

    # Saving original models
    savedir = os.path.join(os.getcwd(), "models")
    os.makedirs(savedir, exist_ok=True)
    torch.save(model_a.state_dict(), os.path.join(savedir, "model_a.pt"))
    torch.save(model_b.state_dict(), os.path.join(savedir, "model_b.pt"))
    torch.save(model_b_original.state_dict(), os.path.join(savedir, "model_b_original.pt"))

    # Check that the model was actually changed
    sum_orig = 0.0
    sum_diff = 0.0
    for (n_old, p_old), (n_new, p_new) in zip(
            model_b_original.named_parameters(), model_b.named_parameters()
    ):
        if "weight" in n_old or "bias" in n_old:
            sum_orig += p_old.abs().sum().item()
            sum_diff += (p_new - p_old).abs().sum().item()

    assert sum_diff / sum_orig > 0.1

    # Get the new accuracy, loss, and perplexity
    acc_br, loss_br, perp_br = eval_full(model_b)

    # Interpolate
    print("Interpolating A - B_Rebasin...")
    savedir_ab_rebasin = os.path.join(os.getcwd(), "models_ab_rebasin")
    os.makedirs(savedir_ab_rebasin, exist_ok=True)
    interp = rebasin.interpolation.LerpSimple(
        models=[model_a, model_b],
        eval_fn=model_loss_full,
        eval_mode="min",
        savedir=savedir_ab_rebasin,
        logging_level="info",
    )
    interp.interpolate(interp_steps)

    print("Interpolating A - B_Original...")
    savedir_ab_orig = os.path.join(os.getcwd(), "models_ab_orig")
    os.makedirs(savedir_ab_orig, exist_ok=True)
    interp = rebasin.interpolation.LerpSimple(
        models=[model_a, model_b_original],
        eval_fn=model_loss_full,
        eval_mode="min",
        savedir=savedir_ab_orig,
        logging_level="info",
    )
    interp.interpolate(interp_steps)

    print("Interpolating B_Original - B_Rebasin...")
    savedir_b_orig_b_rebasin = os.path.join(os.getcwd(), "models_b_orig_b_rebasin")
    os.makedirs(savedir_b_orig_b_rebasin, exist_ok=True)
    interp = rebasin.interpolation.LerpSimple(
        models=[model_b_original, model_b],
        eval_fn=model_loss_full,
        eval_mode="min",
        savedir=savedir_b_orig_b_rebasin,
        logging_level="info",
    )
    interp.interpolate(interp_steps)

    # Evaluate
    print("Evaluating models...")
    print("A-B_Original")
    l_ab_orig, p_ab_orig, a_ab_orig = eval_saved_models(
        savedir_ab_orig
    )
    print("A-B_Rebasin")
    l_ab_rebasin, p_ab_rebasin, a_ab_rebasin = eval_saved_models(
        savedir_ab_rebasin
    )
    print("B_Original-B_Rebasin")
    l_b_orig_b_rebasin, p_b_orig_b_rebasin, a_b_orig_b_rebasin = eval_saved_models(
        savedir_b_orig_b_rebasin
    )

    # Save results
    print("Saving results...")
    losses = {
        "ab_orig": [loss_a] + l_ab_orig + [loss_bo],
        "ab_rebasin": [loss_a] + l_ab_rebasin + [loss_br],
        "b_orig_b_rebasin": [loss_bo] + l_b_orig_b_rebasin + [loss_br]
    }
    perplexities = {
        "ab_orig": [perp_a] + p_ab_orig + [perp_bo],
        "ab_rebasin": [perp_a] + p_ab_rebasin + [perp_br],
        "b_orig_b_rebasin": [perp_bo] + p_b_orig_b_rebasin + [perp_br]
    }
    accuracies = {
        "ab_orig": [acc_a] + a_ab_orig + [acc_bo],
        "ab_rebasin": [acc_a] + a_ab_rebasin + [acc_br],
        "b_orig_b_rebasin": [acc_bo] + a_b_orig_b_rebasin + [acc_br]
    }

    df = pd.DataFrame(losses)
    df.to_csv("losses.csv")
    df = pd.DataFrame(perplexities)
    df.to_csv("perplexities.csv")
    df = pd.DataFrame(accuracies)
    df.to_csv("accuracies.csv")


def test_run_double_num_steps():
    print("\n\nTesting running with double number of steps\n")
    steps_before = hyp['opt']['total_train_steps']
    hyp['opt']['total_train_steps'] *= 2

    model_a, _ = train_model()
    acc_a, loss_a, perp_a = eval_full(model_a)

    # Save results
    with open("results.txt", "w") as f:
        f.write(f"Steps before: {steps_before}\n")
        f.write(f"Steps: {steps_before*2}\n")
        f.write(f"Accuracy: {acc_a}\n")
        f.write(f"Loss: {loss_a}\n")
        f.write(f"Perplexity: {perp_a}\n")

    hyp['opt']['total_train_steps'] = steps_before


def print_model_permutation_structure() -> None:
    input_data, _ = get_batch(
        data, key='eval', batchsize=64, sequence_length=32
    )
    pcd = rebasin.PermutationCoordinateDescent(make_net(), make_net(), input_data)
    print(pcd.pinit.model_graph)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--print", action="store_true", default=False)
    parser.add_argument("-s", "--steps", type=int, default=99)
    parser.add_argument("-d", "--test_double_num_steps", action="store_true", default=False)
    parser.add_argument("-t", "--test_rebasin_package", action="store_true", default=False)
    hparams = parser.parse_args()

    if hparams.print:
        print_model_permutation_structure()
        return

    if hparams.test_rebasin_package:
        test_memory_capacity()
        test_rebasin_works()

    test_rebasin(interp_steps=hparams.steps)

    if hparams.test_double_num_steps:
        test_run_double_num_steps()


if __name__ == "__main__":
    main()