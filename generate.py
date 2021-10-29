# coding=utf-8
# Copyright (c) 2021 Jeffrey M. Binder.  All rights reserved.

from run_generation import *

model_type = 'gpt2'
model_name_or_path = 'gpt2-xl'
device = 'cuda'

length = 100
do_sample = True
temperature = 0.6
k = 5
p = 0.5
repetition_penalty = 1.5
num_return_sequences = 10
overlap_factor = 0.25

prompt_text = '''Scientists recently discovered a new species of {serpent~snake}. Here is a description of it:'''


# Initialize the model and tokenizer
try:
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
except KeyError:
    raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")
tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
model = model_class.from_pretrained(model_name_or_path)
model.to(device)
model.eval()
length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)

import time
start_time = time.time()
output_sequences = model.generate(
    prompt=prompt_text,
    overlap_factor=overlap_factor,
    tokenizer=tokenizer,
    max_length=length,
    temperature=temperature,
    top_k=k,
    top_p=p,
    repetition_penalty=repetition_penalty,
    do_sample=do_sample,
    num_return_sequences=num_return_sequences,
    pad_token_id=0,
    verbose=True,
)
print(f"Time: {time.time() - start_time}s")

# Remove the batch dimension when returning multiple sequences
if len(output_sequences.shape) > 2:
    output_sequences.squeeze_()

generated_sequences = []

for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    generated_sequence = generated_sequence.tolist()
    generated_sequence = [idx for idx in generated_sequence if idx != 0]

    # Decode text
    generated_text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    if num_return_sequences > 1:
        print(f'Generated sequence {generated_sequence_idx}:')
    print(generated_text)

