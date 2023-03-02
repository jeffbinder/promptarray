# coding=utf-8
# Copyright (c) 2021 Jeffrey M. Binder.  All rights reserved.

from generator import PromptArrayGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM

model_type = 'gpt2'
model_name_or_path = 'gpt2-xl'
device = 'cuda'

length = 100
do_sample = True
temperature = 0.6
k = 5
p = 0.5
repetition_penalty = 1.5
bad_words = ["the"]
num_return_sequences = 2
overlap_factor = 0.25

prompt_text = '''Scientists recently discovered a new species of {serpent~snake}. Here is a description of it:'''


# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.to(device)
model.eval()

# Initialize PromptArray
generator = PromptArrayGenerator(
    model,
    tokenizer
)

import time
start_time = time.time()
outputs = generator(
    prompt=prompt_text,
    num_return_sequences=num_return_sequences,
    max_length=length,
    do_sample=do_sample,
    temperature=temperature,
    top_k=k,
    top_p=p,
    repetition_penalty=repetition_penalty,
    bad_words=bad_words,
    overlap_factor=overlap_factor,
    verbose=True
)
print(f"Time: {time.time() - start_time}s")

for i, output in enumerate(outputs):
    if num_return_sequences > 1:
        print(f'Generated sequence {i}:')
    print(output)

