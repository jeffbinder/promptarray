# coding=utf-8
# Copyright (c) 2021 Jeffrey M. Binder.  All rights reserved.

import json
import math
import numpy as np
import nltk
import os
import re
import sys
import torch
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

from program import Program
from generator import PromptArrayGenerator

model_type = 'gpt2'
model_name_or_path = 'gpt2'
device = 'cuda'

test_mode = 'word'

repetition_penalty = None
suppress_punctuation = True
batch_size = 20

prompting_mode = 'sentence' # One of 'default', 'blank', 'fixed', 'word', 'phrase', 'sentence', 'sentence|blank', 'sentence|word', 'sentence|phrase', 'sentence|word|phrase'
prefix = '[...]'
fixed_negative_prompt = '[...] and'
finetune_sentence_tokenizer = False
regularize_text = False
overlap_factor = 0.0

re_phrase_boundary = re.compile('[,.:;?!"“”]')

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.to(device)
model.eval()

generator = PromptArrayGenerator(
    model,
    tokenizer
)

if model_type == 'xlm':
    re_word = re.compile(r"^ ?[A-Za-z']+(</w>)?$")
    re_final_punct = re.compile(r"^.*?([^A-Za-z' ]+(</w>)?)$")
    re_first_punct_on = re.compile(r"^.*?([^@A-Za-z' ]+.*)$")
elif model_type == 'ctrl':
    re_word = re.compile(r"^ ?[@A-Za-z']+$")
    re_final_punct = re.compile(r"^.*s?([^@A-Za-z' ]+)$")
    re_first_punct_on = re.compile(r"^.*?([^@A-Za-z' ]+.*)$")
elif model_type == 'xlnet':
    re_word = re.compile(r"^ ?[A-Za-z'▁]+$")
else:
    re_word = re.compile(r"^ ?[A-Za-z']+$")

if model_type == 'xlm':
    def is_word_piece(idx):
        tok = tokenizer.convert_ids_to_tokens([idx])[0]
        return re_word.match(tok) and not tok.endswith('</w>')
elif model_type == 'ctrl':
    def is_word_piece(idx):
        tok = tokenizer.convert_ids_to_tokens([idx])[0]
        return tok.endswith('@@')
elif model_type == 'xlnet':
    def is_word_piece(idx):
        tok = tokenizer.convert_ids_to_tokens([idx])[0]
        return re_word.match(tok) and not tok.startswith('▁')
else:
    def is_word_piece(idx):
        tok = tokenizer.convert_ids_to_tokens([idx])[0]
        string = tokenizer.convert_tokens_to_string([tok])
        return re_word.match(string) and not string.startswith(' ')
def is_punctuation(idx):
    tok = tokenizer.convert_ids_to_tokens([idx])[0]
    string = tokenizer.convert_tokens_to_string([tok])
    return not re_word.match(string)

punctuation = []
word_pieces = []
vocab = tokenizer.get_vocab()
vocab_size = len(vocab)
for tok in vocab:
    idx = vocab[tok]
    tok = tokenizer.convert_tokens_to_string([tok])
    if not re_word.match(tok):
        punctuation.append([idx])
    if model_type in ('xlm', 'ctrl') and test_mode == 'token' and is_word_piece(idx):
        word_pieces.append([idx])

bos_token = tokenizer.bos_token or tokenizer.cls_token or ''
if model_type == 'ctrl':
    bos_token = 'Books '

# The models have word pieces at the beginning of the word, so we must add in an offset when
# locating word boundaries
if model_type in ('xlm', 'ctrl'):
    word_piece_offset = 1
else:
    word_piece_offset = 0

if model_type in ('xlm', 'ctrl') and test_mode == 'token':
    # Do not allow the prediction of word pieces in token mode because they cannot come at the
    # end of sentence in these models
    bad_words_ids = punctuation.copy() if suppress_punctuation else []
    bad_words_ids += word_pieces
elif model_type in ('openai-gpt', 'gpt2', 'xlnet') and test_mode == 'word':
    # Conversely, with these models, the word pieces come at the end, so they must be suppressed
    # at the beginning when we are trying to predict a word.
    bad_words_ids = punctuation.copy() if suppress_punctuation else []
    bad_words_ids += word_pieces
else:
    bad_words_ids = punctuation if suppress_punctuation else None

fixed_negative_prompt = Program.escape(fixed_negative_prompt)

def run_model(prompt):
    output_sequences = generator(
        prompt=prompt,
        overlap_factor=overlap_factor,
        num_return_sequences=1,
        max_length=1,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        bad_words_ids=bad_words_ids,
        output_token_ids=True,
    )

    if test_mode == 'word':
        # Punctuation is not suppressed after the first token, since it provides one of the ways
        # by which models can decide that the word has ended. The only straightforward way to implement
        # this given how generate() is implemented is to call it twice.
        guess_1 = output_sequences[0, -1]
        tok_1 = tokenizer.decode([guess_1])
        prompt_2 = '{' + prompt + '}' + tok_1
        output_sequences_2 = generator(
            prompt=prompt_2,
            overlap_factor=overlap_factor,
            num_return_sequences=1,
            max_length=5,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            output_token_ids=True,
        )
        output_sequences = torch.cat([output_sequences, output_sequences_2], dim=1)

    if test_mode == 'token':
        guess = output_sequences[0, -1]
        return guess
    else:
        n = output_sequences.shape[1]
        j = 1 - word_piece_offset
        while j < n - word_piece_offset and is_word_piece(output_sequences[0, j]):
            j += 1
        end = j + word_piece_offset
        guess = output_sequences[0, :end].to('cpu')
        return guess

sent_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
if finetune_sentence_tokenizer:
    f = open('../../data/gpt-2/data/lambada_development.jsonl')
    text = []
    text = text.replace('\n', ' ').replace('  ', ' ').replace('“', '"').replace('”', '"').replace('’', '\'').replace('‘', '\'')
    for line in f.readlines():
        text.append(json.loads(line)['text'] + ".")
    text = '\n'.join(text)
    f.close()
    sent_tokenizer.train(text)
def split_last_sentence(text):
    # The following is necessary to get the sentence tokenizer to behave
    regularized_text = text.replace('\n', ' ').replace('  ', ' ').replace('“', '"').replace('”', '"').replace('’', '\'').replace('‘', '\'')
    sentences = sent_tokenizer.tokenize(regularized_text)
    n = len(sentences[-1])
    return text[:-(n+1)], text[-n:]

def interpret_line(line):
    text = json.loads(line)['text']
    if regularize_text:
        text = text.replace('\n', ' ').replace('  ', ' ').replace('“', '"').replace('”', '"').replace('’', '\'').replace('‘', '\'')

    # Separate the prompt from the desired output
    ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    if test_mode == 'token':
        prompt = ids[0,:-1]
        answer = ids[0,-1]
    else:
        n = ids.shape[1]
        i = 1 + word_piece_offset
        while i <= n:
            if not is_word_piece(ids[0,-i]):
                break
            i += 1
        i -= word_piece_offset
        prompt = ids[0,:-i]
        answer = ids[0,-i:]
    prompt = tokenizer.decode(prompt)
    prompt = Program.escape(prompt)

    if prompting_mode == 'default':
        pass

    elif prompting_mode == 'blank':
        prompt = f'{prompt}~'

    elif prompting_mode == 'fixed':
        prompt = f'{prompt}~{fixed_negative_prompt}'

    elif prompting_mode == 'word':
        toks = nltk.word_tokenize(prompt)
        last_tok = Program.escape(toks[-1])
        prompt = f'{prompt}~{prefix}{last_tok}'

    elif prompting_mode == 'phrase':
        phrases = re_phrase_boundary.split(prompt)
        last_phrase = Program.escape(phrases[-1])
        prompt = f'{prompt}~{prefix}{last_phrase}'
    
    elif prompting_mode == 'sentence':
        first_sentences, last_sentence = split_last_sentence(prompt)
        last_sentence = Program.escape(last_sentence)
        prompt = f'{prompt}~{prefix}{last_sentence}'
    
    elif prompting_mode == 'sentence|blank':
        first_sentences, last_sentence = split_last_sentence(prompt)
        last_sentence = Program.escape(last_sentence)
        prompt = f'{prompt}~{prefix}{{{last_sentence}|}}'

    elif prompting_mode == 'sentence|word':
        _, last_sentence = split_last_sentence(prompt)
        last_sentence = Program.escape(last_sentence)
        toks = nltk.word_tokenize(prompt)
        last_tok = Program.escape(toks[-1])
        prompt = f'{prompt}~{prefix}{{{last_sentence}|{last_tok}}}'

    elif prompting_mode == 'sentence|phrase':
        _, last_sentence = split_last_sentence(prompt)
        last_sentence = Program.escape(last_sentence)
        phrases = re_phrase_boundary.split(prompt)
        last_phrase = Program.escape(phrases[-1])
        prompt = f'{prompt}~{prefix}{{{last_sentence}|{last_phrase}}}'

    elif prompting_mode == 'sentence|word|phrase':
        _, last_sentence = split_last_sentence(prompt)
        last_sentence = Program.escape(last_sentence)
        toks = nltk.word_tokenize(prompt)
        last_tok = Program.escape(toks[-1])
        phrases = re_phrase_boundary.split(prompt)
        last_phrase = Program.escape(phrases[-1])
        prompt = f'{prompt}~{prefix}{{{last_sentence}|{last_tok}|{last_phrase}}}'

    else:
        raise ValueError("Unknown prompting mode!")

    return text, prompt, answer

f = open('../../data/gpt-2/data/lambada_test.jsonl')
total_score = 0.0
texts = []
prompts = []
answers = []
for line in f.readlines():
    text, prompt, answer = interpret_line(line)
    texts.append(text)
    prompts.append(prompt)
    answers.append(answer)

n = 0
ncorrect = 0
for text, prompt, answer in zip(texts, prompts, answers):
    guess = run_model(prompt)
    n += 1

    if model_type == 'ctrl' and test_mode == 'token':
        guess = [guess]
        answer = [answer]

    if test_mode == 'token':
        if model_type in ('xlm', 'ctrl'):
            guess_text = tokenizer.decode(guess)
            m = re_final_punct.match(guess_text)
            if m:
                guess_text = guess_text[:-len(m.group(1))]
            answer_text = tokenizer.decode(answer)
            correct = guess_text == answer_text
        else:
            correct = guess == answer
    else:
        if model_type in ('xlm', 'ctrl'):
            guess_text = tokenizer.decode(guess)
            m = re_first_punct_on.match(guess_text)
            if m:
                guess_text = guess_text[:-len(m.group(1))]
            answer_text = tokenizer.decode(answer)
            correct = guess_text == answer_text
        else:
            correct = guess.equal(answer)
    if correct:
        ncorrect += 1
    
    if n % 100 == 0:
        guess = tokenizer.decode(guess)
        print('----------')
        print(f'Text: {text}')
        print(f'Guess: {guess} - {"correct" if correct else "wrong"} ({ncorrect}/{n} = {100*ncorrect/n})')

print(f'Final results: {ncorrect}/{n} = {100*ncorrect/n}')
