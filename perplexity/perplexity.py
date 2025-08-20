from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from accelerate.test_utils.testing import get_backend
import os
import json
import sys

import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
args = parser.parse_args()

# load all jsonl files in the directory

original_descriptions = []
new_descriptions = []

for file in os.listdir(args.dir):
    if file.endswith("-improvement_history.jsonl"):
        with open(os.path.join(args.dir, file), "r") as f:
            best_improvement = None
            best_percent = 0
            for line in f:
                data = json.loads(line)
                if data['attempt'] == 0:
                    original_descriptions.append(data['description'])
                else:
                    if data['percent'] >= best_percent:
                        best_percent = data['percent']
                        best_improvement = data['improvement']
            if best_improvement is not None:
                new_descriptions.append(best_improvement)

# print(original_descriptions)
# print(new_descriptions)
# sys.exit()
                    



device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
model_id = "openai-community/gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

def get_ppl(encodings, max_length, stride, model):
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    seq_len = encodings.input_ids.size(1)
    for begin_loc in tqdm(range(0, seq_len, stride)):
        print(f"begin_loc: {begin_loc}, seq_len: {seq_len}")
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return ppl


max_length = model.config.n_positions
stride = 1

# Store perplexity and sequence length together
orig_data = []  # List of tuples (perplexity, sequence_length)
new_data = []   # List of tuples (perplexity, sequence_length)

print("Computing perplexity and sequence length for original descriptions...")
for orig_desc in tqdm(original_descriptions, desc="Original"):
    encodings = tokenizer(orig_desc, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    ppl = get_ppl(encodings, max_length, stride, model)
    orig_data.append((ppl.item(), seq_len))

print("Computing perplexity and sequence length for new descriptions...")
for new_desc in tqdm(new_descriptions, desc="New"):
    encodings = tokenizer(new_desc, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    ppl = get_ppl(encodings, max_length, stride, model)
    new_data.append((ppl.item(), seq_len))

# Extract perplexity and sequence length for plotting
orig_ppls, orig_seq_lens = zip(*orig_data) if orig_data else ([], [])
new_ppls, new_seq_lens = zip(*new_data) if new_data else ([], [])

# Plotting

plt.figure(figsize=(8, 6))
plt.scatter(torch.log(torch.tensor(orig_ppls)), torch.log(torch.tensor(orig_seq_lens)), color='blue', label='Original Descriptions', alpha=0.7)
plt.scatter(torch.log(torch.tensor(new_ppls)), torch.log(torch.tensor(new_seq_lens)), color='orange', label='Attacked Descriptions', alpha=0.7)
plt.xlabel('Log Perplexity')
plt.ylabel('Log Sequence Length')
plt.title(f'GPT2 Log Perplexity vs Log Sequence Length for {args.dir}')
plt.legend()
plt.tight_layout()

# Save scatter plot to the specified directory
plot_path = 'perplexity_vs_sequence_length.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Scatter plot saved to: {plot_path}")
plt.close()  # Close the figure to free memory

# Create histogram plot of log-perplexity values
plt.figure(figsize=(10, 6))
log_orig_ppls = torch.log(torch.tensor(orig_ppls))
log_new_ppls = torch.log(torch.tensor(new_ppls))
plt.hist(log_orig_ppls, bins=30, alpha=0.7, color='blue', label='Original Descriptions')
plt.hist(log_new_ppls, bins=30, alpha=0.7, color='orange', label='Attacked Descriptions')
plt.xlabel('Log Perplexity')
plt.ylabel('Count')
plt.title(f'GPT2 Log Perplexity Distribution for {args.dir}')
plt.legend()
plt.tight_layout()

# Save histogram plot
hist_plot_path = 'perplexity_histogram.png'
plt.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
print(f"Histogram plot saved to: {hist_plot_path}")
plt.close()  # Close the figure to free memory

# Print summary statistics
print(f"\nOriginal descriptions: {len(orig_data)} samples")
print(f"New descriptions: {len(new_data)} samples")
if orig_data:
    print(f"Original - Avg PPL: {sum(orig_ppls)/len(orig_ppls):.2f}, Avg Seq Len: {sum(orig_seq_lens)/len(orig_seq_lens):.1f}")
if new_data:
    print(f"New - Avg PPL: {sum(new_ppls)/len(new_ppls):.2f}, Avg Seq Len: {sum(new_seq_lens)/len(new_seq_lens):.1f}")


