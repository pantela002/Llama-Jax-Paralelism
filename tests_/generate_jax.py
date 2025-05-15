import os
import jax
import jax.numpy as jnp
import numpy as np
import fire
from flax.core.frozen_dict import freeze
from jax_llama import FlaxLLaMAForCausalLM, convert_llama_weights
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from jax_llama.generation import LLaMA  # your class is here

def jax_load(ckpt_dir: str, tokenizer_path: str, max_seq_length: int = 2048) -> LLaMA:
    print("🔧 Loading tokenizer and weights...")
    tokenizer = LLaMA3Tokenizer(tokenizer_path)

    jax_params, jax_config = convert_llama_weights(
        ckpt_dir=ckpt_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length
    )
    jax_params = freeze(jax.tree.map(jnp.asarray, jax_params))

    model = FlaxLLaMAForCausalLM(config=jax_config, _do_init=False)
    llama = LLaMA(params=jax_params, model=model, tokenizer=tokenizer)

    return llama

def main(
    ckpt_dir: str = "/root/tt/sw/llama3.1-8B/8B",
    tokenizer_path: str = "/root/tt/sw/llama3.1-8B/original/tokenizer.model",
    prompt: str = (
    "Q: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?\n"
    "A: The cost of the house and repairs came out to 80,000 + 50,000 = $<<80000+50000=130000>>130,000\n"
    "He increased the value of the house by 80,000 * 1.5 = <<80000*1.5=120000>>120,000\n"
    "So the new value of the house is 120,000 + 80,000 = $<<120000+80000=200000>>200,000\n"
    "So he made a profit of 200,000 - 130,000 = $<<200000-130000=70000>>70,000\n"
    "F: #### 70000\n\n"
    "Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\n"
    "A:"
),
    max_gen_len: int = 128,
    temperature: float = 0.1,
    top_p: float = 0.99
):
    print("🚀 Loading LLaMA...")
    llama = jax_load(ckpt_dir, tokenizer_path)

    print("✍️ Generating...")
    results = llama.generate_from_str(
        [prompt],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for i, r in enumerate(results):
        print(f"\n🧾 Prompt {i + 1}: {prompt}")
        print("🧠 Output:", r)

if __name__ == "__main__":
    fire.Fire(main)
