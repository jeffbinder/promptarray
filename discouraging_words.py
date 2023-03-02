# coding=utf-8
# Copyright (c) 2021 Jeffrey M. Binder.  All rights reserved.

import codecs
import nltk
import random
import scipy.stats
import torch

from generator import PromptArrayGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM

model_type = 'gpt2'
model_name_or_path = 'gpt2-xl'
device = 'cuda'

length = 300
do_sample = True
temperature = 0.6
top_k = 5
top_p = 0.5
repetition_penalty = 1.5
num_return_sequences = 10
num_batches = 100
seed = 14891435220765460437

experiment_name = "snake~legs"
prompt_v1 = '''Scientists recently discovered a new species of snake. Here is a description of it:'''
prompt_v2 = '''Scientists recently discovered a new species of snake{~ with legs}. Here is a description of it:'''
words_to_count = [("leg", "legs", "legged"), ("fur", "furred", "furry"), ("hair", "hairs", "haired", "hairy")]
barnard_test_alternative = "greater"

f1 = codecs.open(f"discouraging-results/{experiment_name}-v1", "w", "utf8")
f2 = codecs.open(f"discouraging-results/{experiment_name}-v2", "w", "utf8")

# Initialize the model and tokenizer
torch.manual_seed(seed)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.to(device)
model.eval()

# Initialize PromptArray
generator = PromptArrayGenerator(
    model,
    tokenizer
)

counts_1 = {word: 0 for word in words_to_count}
counts_2 = {word: 0 for word in words_to_count}
for batch_num in range(num_batches):
    print(f"Batch {batch_num}")
    for i, prompt in enumerate([prompt_v1, prompt_v2]):
        # Needed to avoid weirdness with Torch's random number generator
        output_sequences = generator(
            prompt=prompt,
            max_length=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            output_token_ids=True,
            verbose=True,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()
            generated_sequence = [idx for idx in generated_sequence if idx != 0]
            idx = generated_sequence_idx + batch_num * num_return_sequences

            # Decode text
            generated_text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            f = f1 if i == 0 else f2
            f.write(f"Sequence {idx}:\n{generated_text}\n\n----------\n")

            counts = counts_1 if i == 0 else counts_2
            toks = [tok.lower() for tok in nltk.word_tokenize(generated_text)]
            for word in words_to_count:
                variant_found = False
                for variant in word:
                    if variant in toks:
                        variant_found = True
                        break
                if variant_found:
                    counts[word] += 1

    f1.flush()
    f2.flush()
    print("word\tv1\tv2")
    n = num_return_sequences * (batch_num+1)
    for word in words_to_count:
        x1 = counts_1[word]
        x2 = counts_2[word]
        print(f"{word}\t{x1}/{n}\t{x2}/{n}")

print("word\tv1\tv2\tp")
n = num_return_sequences * num_batches
for word in words_to_count:
    x1 = counts_1[word]
    x2 = counts_2[word]
    o = scipy.stats.barnard_exact([[x1, x2], [n-x1, n-x2]], alternative=barnard_test_alternative)
    p = o.pvalue
    print(f"{word}\t{x1}/{n}\t{x2}/{n}\t{p}")

f1.close()
f2.close()

