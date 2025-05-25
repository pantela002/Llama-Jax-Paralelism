
import jax
import jax.numpy as jnp
import fire
from jax_llama import FlaxLLaMAForCausalLM, convert_llama_weights
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from jax_llama.generation import LLaMA  # your class is here
from flax.traverse_util import flatten_dict
import jax.debug
from jax_llama.config import device_mesh
from flax.traverse_util import flatten_dict
from jax.sharding import PartitionSpec as P, NamedSharding
import flax.nnx as nnx

def manually_shard_params(params, mesh):
    with mesh:
        for path, value in nnx.items(params):
            # Only shard 2D weights named 'kernel'
            if path[-1] == "kernel" and value.ndim >= 2:
                print(f"✅ Sharding {path} with shape {value.shape}")
                sharded = jax.device_put(value, NamedSharding(mesh, P(None, "mp")))
                nnx.set(params, path, sharded)
            elif path[-1] == "kernel":
                print(f"⚠️ Skipping {path} with shape {value.shape}")
    return params


def jax_load(ckpt_dir: str, tokenizer_path: str, mesh, max_seq_length: int = 2048) -> LLaMA:
    print("🔧 Loading tokenizer and weights...")
    tokenizer = LLaMA3Tokenizer(tokenizer_path)

    # Convert weights to dict
    params_np, jax_config = convert_llama_weights(
        ckpt_dir=ckpt_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
    )

    # Instantiate model without initializing
    model = FlaxLLaMAForCausalLM(config=jax_config, _do_init=False)

    # Build an empty state from module structure
    empty_params = nnx.state_from_module(model.module)

    # Load weights from params_np into empty state
    for path, _ in nnx.items(empty_params.params):
        if path in params_np:
            nnx.set(empty_params.params, path, jnp.asarray(params_np[path]))

    # Manually shard kernel weights under mesh
    with mesh:
        for path, value in nnx.items(empty_params.params):
            if path[-1] == "kernel" and value.ndim >= 2:
                print(f"✅ Sharding {path} with shape {value.shape}")
                sharded = jax.device_put(value, NamedSharding(mesh, P(None, "mp")))
                nnx.set(empty_params.params, path, sharded)

    # Inject the loaded + sharded state
    model.module.state = empty_params

    # Return wrapped model
    return LLaMA(params=empty_params.params, model=model, tokenizer=tokenizer, mesh=mesh)


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
    max_gen_len: int = 16,
    temperature: float = 0.1,
    top_p: float = 0.99
):
    # print mesh
    print("✅ Mesh initialized:", device_mesh)

    print("🚀 Loading LLaMA...")
    llama = jax_load(ckpt_dir, tokenizer_path, mesh=device_mesh)

    print("\n🔍 Visualizing sharded parameter placements (first few):")
    flat_params = flatten_dict(llama.params)
    for k, v in list(flat_params.items())[:5]:
        print(f"🔹 Param {k} sharding:")
        jax.debug.visualize_array_sharding(v)


    print("✍️ Generating...")
    with device_mesh:
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
