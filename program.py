# coding=utf-8
# Copyright (c) 2021 Jeffrey M. Binder.  All rights reserved.

import torch
from torch.nn import functional as F
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from lark import Lark

parser = Lark(r'''
start: _exp1

_exp1: or_exp | _exp2
_exp2: and_exp | _exp3
_exp3: not_exp | _exp4
_exp4: concat_exp | _exp5
_exp5: "{" _exp1 "}" | text

or_exp: _exp2 "|" _exp1
and_exp: _exp3 "&" _exp2
not_exp: _exp4 "~" _exp3
concat_exp: _exp5 _exp4

text: /([^|&~{}\\]|\\[|&~{}\\])+/?
''')

class Operation():
    def __init__(self, type: str, arg1: int, arg2: Optional[int] = None):
        self.type = type
        self.arg1 = arg1
        self.arg2 = arg2

    def __str__(self):
        return f"{self.arg1} {self.type}= {self.arg2}"

    def __call__(self, tensor: torch.Tensor, num_return_sequences: int):
        for i in range(num_return_sequences):
            arg1 = self.arg1*num_return_sequences + i
            arg2 = self.arg2*num_return_sequences + i

            if self.type == '|':
                tensor[arg1, :] += tensor[arg2, :]
            elif self.type == '&':
                tensor[arg1, :] *= tensor[arg2, :]
            elif self.type == '~':
                tensor[arg1, :] /= torch.sqrt(tensor[arg2, :])
        
        # nan appears when scores are set to -inf by preprocessors--replace with -inf to keep these
        # tokens from being selected
        nan_mask = tensor.isnan()
        tensor.masked_fill_(nan_mask, -float("inf"))

        return tensor

    def shift(self, k: int):
        return Operation(self.type, self.arg1 + k, None if self.arg2 is None else self.arg2 + k)

class Program():
    def __init__(self):
        self.ops = []
        
    def __call__(
        self,
        logits: torch.tensor,
        num_return_sequences: int
    ) -> Tuple[List[torch.LongTensor], List[Operation]]:
        """
        Executes the opcode produced by the compiler.
        """

        # run the code!
        logits = F.softmax(logits, dim=-1)
        for op in self.ops:
            logits = op(logits, num_return_sequences)
        
        # replace the results for all variants with the combined result
        num_variants = logits.shape[0] // num_return_sequences
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
        verbose: bool
    ):
        """
        Parses a Boolean prompt and transforms it into a form suitable for use with the model. Returns two things: 1) a list of tensors comprising all combinations of tokens permissible based on the Boolean expression, suitable for use as input to the model; and 2) a list of operations that must be performed to combine the output logits based on the expression.
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
            elif node.data in ('or_exp', 'and_exp', 'not_exp'):
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
                elif node.data == 'not_exp':
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

        program = Program()
        program.ops = ops

        return program, ids, attention_mask

    @classmethod
    def escape(self, prompt: str):
        r"""
        Escapes special characters in a string for use in a Boolean prompt.
        """
        escaped_prompt = ""
        for c in prompt:
            if c in "\\|&~{}":
                escaped_prompt += "\\"
            escaped_prompt += c
        return escaped_prompt
    