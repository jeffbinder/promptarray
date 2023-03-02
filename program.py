# coding=utf-8
# Copyright (c) 2021-23 Jeffrey M. Binder.  All rights reserved.

import math
import time
import torch
from torch.nn import functional as F

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from lark import Lark

parser = Lark(r'''
start: _exp1

_exp1: or_exp | sub_exp | _exp2
_exp2: and_exp | div_exp | _exp3
_exp3: rather_exp | _exp4
_exp4: concat_exp | _exp5
_exp5: "{" _exp1 "}" | text

or_exp: _exp1 "|" _exp2
sub_exp: _exp1 "^" _exp2
and_exp: _exp2 "&" _exp3
div_exp: _exp2 "/" _exp3
rather_exp: _exp3 "~" _exp4
concat_exp: _exp5 _exp4

text: /([^\/|&~{}^\\]|\\[\/|&~{}^\\])+/?
''')

class ProgramConfig():
    def __init__(self, vocab_size: int, overlap_factor: float):
        self.vocab_size = vocab_size
        self.overlap_factor = overlap_factor

class Operation():
    def __init__(self, type: str, *args: Tuple[int]):
        self.type = type
        self.args = list(args)

    def __str__(self):
        arg_str = ', '.join([str(a) for a in self.args[1:]])
        return f"{self.args[0]} {self.type}= {arg_str}"

    def add_arg(self, arg: int):
        self.args.append(arg)

    def shift(self, k: int):
        return Operation(self.type, *[arg + k for arg in self.args])

    def __call__(self, config: ProgramConfig, logits: torch.Tensor, num_return_sequences: int):
        for i in range(num_return_sequences):
            args = [arg*num_return_sequences + i for arg in self.args]
            t0 = logits[args[0], :]
            t1 = logits[args[1], :]

            if self.type == '|':
                t_and = t0 * t1
                t_and /= torch.sum(t_and)
                p = config.overlap_factor
                t0 = (t0 + t1 - p*t_and) / (2 - p)

            elif self.type == '&':
                t0 = t0 * t1
                t0 /= torch.sum(t0)

            elif self.type == '^':
                t0 *= 1 - t1/torch.max(t1)
                t0 /= torch.sum(t0)

            elif self.type == '/':
                t0 /= t1
                t0 /= torch.sum(t0)

            elif self.type == '~':
                t0 *= t0 / t1
                t0 /= torch.sum(t0)

            logits[args[0], :] = t0
        
        nan_mask = logits.isnan()
        logits.masked_fill_(nan_mask, -float("inf"))

        return logits

class Program():
    def __init__(self, prompt: str, vocab_size: int, overlap_factor: float):
        self.prompt = prompt
        self.ops = []
        self.config = ProgramConfig(vocab_size, overlap_factor)
        
    def __call__(
        self,
        logits: torch.tensor,
        num_return_sequences: int
    ) -> Tuple[List[torch.LongTensor], List[Operation]]:
        """
        Executes the opcode produced by the compiler.
        """
        num_variants = logits.shape[0] // num_return_sequences

        # run the code!
        for op in self.ops:
            logits = op(
                self.config, logits, num_return_sequences
            )
        
        # replace the results for all variants with the combined result
        for j in range(num_return_sequences):
            for k in range(1, num_variants):
                logits[j + k * num_return_sequences, :] = logits[j, :]
        
        return logits

    @classmethod
    def compile(
        cls,
        prompt: str,
        tokenizer: Any,
        bos_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        overlap_factor: float,
        verbose: bool
    ):
        """
        Parses a Boolean prompt and transforms it into a form suitable for use with the model. Returns three things: 1) a Program object that can be called to execute the operations included in the prompt, 2) a tensor comprising all possible prompt variants, suitable for use as input to the model; and 3) an attention mask that must be used when the model is run, so as to account for prompt variants of different lengths.
        """
        tree = parser.parse(prompt)

        # Depth-first traversal
        def _dft(node, strings):
            if node.data == 'start':
                return _dft(node.children[0], strings)
            elif node.data == 'concat_exp':
                ops = []
                for child in node.children:
                    nstrings = len(strings)
                    strings, child_ops = _dft(child, strings)
                    multiplier = len(strings) // nstrings
                    multiplied_ops = []
                    for k in range(multiplier):
                        multiplied_ops += [op.shift(k*nstrings) for op in ops]
                    ops = multiplied_ops + child_ops
                return strings, ops
            elif node.data in ('or_exp', 'and_exp', 'rather_exp', 'sub_exp', 'div_exp'):
                child1_strings, child1_ops = _dft(node.children[0], strings)
                shift = len(child1_strings)
                new_strings = child1_strings
                ops = child1_ops
                child2_strings, child2_ops = _dft(node.children[1], strings)
                new_strings += child2_strings
                ops += [op.shift(shift) for op in child2_ops]
                if node.data == 'or_exp':
                    optype = '|'
                elif node.data == 'and_exp':
                    optype = '&'
                elif node.data == 'sub_exp':
                    optype = '^'
                elif node.data == 'div_exp':
                    optype = '/'
                elif node.data == 'rather_exp':
                    optype = '~'
                ops.append(Operation(optype, 0, shift))
                return new_strings, ops
            elif node.data == 'text':
                if node.children:
                    txt = node.children[0].value
                else:
                    txt = ""
                unescaped_txt = ''
                escape = False
                for c in txt:
                    if not escape and c == '\\':
                        escape = True
                    else:
                        unescaped_txt += c
                        escape = False
                txt = unescaped_txt
                return [s + txt for s in strings], []
        strings, ops = _dft(tree, [""])

        if verbose:
            print("-- Generated prompt variants:")
            for i, s in enumerate(strings):
                print(f"{i}: {s}")
            print("-- Program:")
            if not ops:
                print("[No operations]")
            for i, op in enumerate(ops):
                print(f"{i}: {op}")

        input_ids = []
        max_len = 0
        for s in strings:
            toks = tokenizer.tokenize(s)
            ids = tokenizer.convert_tokens_to_ids(toks)
            if bos_token_id is not None:
                ids = [bos_token_id] + ids
            input_ids.append(ids)
            if len(ids) > max_len:
                max_len = len(ids)
                
        input_ids_padded = []
        input_attention_mask = []
        for ids in input_ids:
            n_pad_toks = max_len - len(ids)
            padded_ids = [pad_token_id or 0] * n_pad_toks + ids
            input_ids_padded.append(padded_ids)
            attention_mask = [0] * n_pad_toks + [1] * len(ids)
            input_attention_mask.append(attention_mask)
        
        ids = torch.tensor(input_ids_padded)
        attention_mask = torch.tensor(input_attention_mask)

        program = Program(prompt, vocab_size, overlap_factor)
        program.ops = ops

        return program, ids, attention_mask

    @classmethod
    def escape(self, prompt: str):
        r"""
        Escapes special characters in a string for use in a Boolean prompt.
        """
        escaped_prompt = ""
        for c in prompt:
            if c in "\\/|&~^{}":
                escaped_prompt += "\\"
            escaped_prompt += c
        return escaped_prompt
    