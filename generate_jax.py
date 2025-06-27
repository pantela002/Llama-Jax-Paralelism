import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax
import jax.numpy as jnp
import numpy as np
import fire
from flax.core.frozen_dict import freeze
from jax_llama import FlaxLLaMAForCausalLM, convert_llama_weights
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from jax_llama.generation import LLaMA  # your class is here
from jax.sharding import Mesh
from flax.traverse_util import flatten_dict
import jax.debug
from jax.experimental import pjit as old_pjit
import gc

def jax_load(ckpt_dir: str, tokenizer_path: str, mesh, max_seq_length: int = 2048) -> LLaMA:
    print("🔧 Loading tokenizer and weights...")
    tokenizer = LLaMA3Tokenizer(tokenizer_path)

    params_np, jax_config = convert_llama_weights(
        ckpt_dir=ckpt_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length
    )
    jax_params = freeze(jax.tree.map(jnp.asarray, params_np))

    del params_np
    gc.collect()    

    model = FlaxLLaMAForCausalLM(config=jax_config, _do_init=False)
    llama = LLaMA(params=jax_params, model=model, tokenizer=tokenizer, mesh=mesh)
    del jax_params
    gc.collect()
    return llama

def main(
    ckpt_dir: str = "/root/tt/3_1_8b/Llama-Jax-Paralelism/llama3.1-8B/8B",
    tokenizer_path: str = "/root/tt/3_1_8b/Llama-Jax-Paralelism/llama3.1-8B/original/tokenizer.model",
    prompt = (
        "Q: A bumper car rink has 12 red cars. They have 2 fewer green cars than they have red cars. "
        "They have 3 times the number of blue cars as they have green cars. The rink also has yellow cars. "
        "If the rink has 75 cars in total how many yellow cars do they have?\n"
        "A:"
    ),
    max_gen_len: int = 64,
    temperature: float = 1,
    top_p: float = 1
):
    # Define mesh
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('mp',))    
    print("✅ Mesh initialized:", mesh)

    print("🚀 Loading LLaMA...")
    llama = jax_load(ckpt_dir, tokenizer_path, mesh=mesh)

    print("✍️ Generating...")
    with mesh:
        results = llama.generate_from_str(
            [prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
    del llama
    gc.collect()
    print("✅ Generation complete.")
    np.savetxt("output_jax.txt", results, fmt="%d")
    #for i, r in enumerate(results):
    #    print(f"\n🧾 Prompt {i + 1}: {prompt}")
    #    print("🧠 Output:", r)

if __name__ == "__main__":
    fire.Fire(main)
