# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from llama import Llama



ckpt_dir = './llama-2-7b'
tokenizer_path = './llama_tok'
temperature: float = 0.6
top_p: float = 0.9
max_seq_len: int = 128
max_gen_len: int = 64
max_batch_size: int = 4

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

print(generator.tokenizer.encode('something', True, True))