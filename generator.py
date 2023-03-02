# coding=utf-8
# Copyright (c) 2021-23 Jeffrey M. Binder.  All rights reserved.
#
# This file contains a few lines of code adapted from the HuggingFace Transformers library,
# which is under the Apache license. This library is covered by the following copyright statement:
#
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import transformers

from program import Program

class PromptArrayGenerator:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: Any,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: Optional[bool] = True
    ):
        self.model = model
        self.vocab_size = self.model.config.vocab_size
        self.bos_token_id = bos_token_id or self.model.config.bos_token_id
        self.pad_token_id = pad_token_id or self.model.config.pad_token_id or 0
        self.eos_token_id = eos_token_id or self.model.config.eos_token_id
        self.device = self.model.device

        self.tokenizer = tokenizer
        self.use_cache = use_cache

    def __call__(
        self,
        prompt: str,
        num_return_sequences: int = 1,
        max_length: int = None,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words: Optional[List[str]] = None,
        bad_words_ids: Optional[List[List[int]]] = None,
        overlap_factor: float = 0.25,
        output_token_ids: bool = False,
        verbose: bool = False
    ):
        if bad_words and bad_words_ids:
            raise ValueError("Cannot specify both `bad_words` and `bad_words_ids`!")
        elif bad_words:
            bad_words_ids = [self.tokenizer.encode(s) for s in bad_words]
            bad_words_ids += [self.tokenizer.encode(" " + s) for s in bad_words]
            bad_words_ids += [self.tokenizer.encode(s.title()) for s in bad_words]
            bad_words_ids += [self.tokenizer.encode(" " + s.title()) for s in bad_words]

        model_kwargs = {
            "use_cache": self.use_cache
        }

        with torch.no_grad():
            program, input_ids, attention_mask = Program.compile(
                prompt,
                self.tokenizer,
                self.bos_token_id,
                self.pad_token_id,
                self.vocab_size,
                overlap_factor,
                verbose
            )
            
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
            input_ids = input_ids.to(self.device)
            model_kwargs["attention_mask"] = attention_mask.to(self.device)

            single_token_bad_words = []
            multitoken_bad_words = []
            if bad_words_ids:
                for word in bad_words_ids:
                    if len(word) == 1:
                        single_token_bad_words.append(word[0])
                    else:
                        multitoken_bad_words.append(word)
                single_token_bad_words_mask = torch.zeros(self.model.config.vocab_size)
                single_token_bad_words_mask[single_token_bad_words] = 1
                single_token_bad_words_mask = single_token_bad_words_mask.unsqueeze(0).to(input_ids.device).bool()

            unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
            num_variants = input_ids.shape[0] // num_return_sequences
            prompt_len = input_ids.shape[-1]
            cur_length = 0

            while cur_length < max_length:

                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = self.model(**model_inputs, return_dict=True)

                scores = outputs.logits[:, -1, :]
                scores = torch.nn.functional.softmax(scores, dim=-1)
                scores = program(scores, num_return_sequences)

                if temperature is not None:
                    scores = scores / temperature

                if top_k is not None:
                    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
                    scores = scores.masked_fill(indices_to_remove, -float("Inf"))

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
                    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    scores = scores.masked_fill(indices_to_remove, -float("Inf"))

                if repetition_penalty is not None:
                    score = torch.gather(scores, 1, input_ids)
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    scores.scatter_(1, input_ids, score)

                if bad_words_ids:
                    bad_words_mask = single_token_bad_words_mask.clone()
                    for banned_token_seq in multitoken_bad_words:
                        prev_tokens = banned_token_seq[:-1]
                        prev_tokens_length = len(prev_tokens)
                        if len(input_ids) >= prev_tokens_length and input_ids[-prev_tokens_length:] == banned_token_seq[:-1]:
                            bad_words_mask[banned_token_seq[-1]] = 1
                    scores = scores.masked_fill(bad_words_mask, -float("Inf"))

                if do_sample:
                    probs = torch.nn.functional.softmax(scores[:num_return_sequences, :], dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1).repeat(num_variants)
                else:
                    next_tokens = torch.argmax(scores, dim=-1)

                next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        model_kwargs["attention_mask"],
                        model_kwargs["attention_mask"].new_ones((attention_mask.shape[0], 1))
                    ],
                    dim=-1
                )

                cur_length += 1
                unfinished_sequences = unfinished_sequences.mul((next_tokens != self.eos_token_id).long())
                if unfinished_sequences.max() == 0:
                    break

            output_ids = input_ids[0:num_return_sequences, prompt_len:]
            if len(output_ids.shape) > 2:
                output_ids.squeeze_()
            
            if output_token_ids:
                return output_ids
            else:
                text_outputs = [
                    self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                    for generated_sequence in output_ids
                ]
                return text_outputs

