# coding=utf-8
# Copyright (c) 2021 Jeffrey M. Binder.  All rights reserved.

from generator import PromptArrayGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = 'openai/gpt-oss-20b'
device = 'cuda'

length = 2000
do_sample = True
temperature = 0.3
k = 10
p = 1.0
repetition_penalty = 2.0
bad_words = None
num_return_sequences = 2
overlap_factor = 0.25
chat_mode = True
chat_mode_think_first = True
chat_mode_max_thought_length = 2000

prompt_text = '''Write an imaginary description of a new species of {serpent~snake}.'''


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
    chat_mode=chat_mode,
    chat_mode_think_first=chat_mode_think_first,
    chat_mode_max_thought_length=chat_mode_max_thought_length,
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

