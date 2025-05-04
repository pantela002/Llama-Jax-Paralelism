import jax
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze
from jax_llama import convert_llama_weights, LLaMA, FlaxLLaMAForCausalLM, get_llama_param_partition_spec, LLaMA3Tokenizer
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
import fire
import os
import numpy as np

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

def load(ckpt_dir: str, tokenizer_path: str, max_seq_length: int=2048, **model_kwargs) -> LLaMA:
    # setup jax mesh
    #devices = mesh_utils.create_device_mesh((1, len(jax.devices())))
    if len(jax.devices()) == 1:
        devices = np.array(jax.devices()).reshape(1, 1)
        mesh = Mesh(devices, axis_names=('dp', 'mp'))
    else:
        devices = np.array(jax.devices()).reshape(2, 4)
        mesh = Mesh(devices, axis_names=('dp', 'mp'))

    print("📌 Mesh setup:", mesh)
    
    # load jax model
    tokenizer = LLaMA3Tokenizer(tokenizer_path)
    jax_params, jax_config = convert_llama_weights(ckpt_dir, tokenizer, max_seq_len=max_seq_length)
    jax_params = freeze(jax.tree.map(jnp.asarray, jax_params))

    # Partition spec
    param_spec = freeze(get_llama_param_partition_spec(unfreeze(jax_params)))
    jax_params = jax.tree_util.tree_map(
        lambda param, spec: jax.device_put(param, NamedSharding(mesh, spec)),
        jax_params, param_spec
    )

    # 🧠 (Optional) Visualize where your params live
    from flax.traverse_util import flatten_dict
    print("\n🔍 Example sharding:")
    flat = flatten_dict(unfreeze(jax_params))
    for k, v in list(flat.items())[:5]:  # show 5 keys
        print(f"{k}: {v.sharding}")

    model = FlaxLLaMAForCausalLM(jax_config, _do_init=False)
    return LLaMA(jax_params, model, tokenizer, mesh=mesh)

def main(ckpt_dir: str, tokenizer_path: str, is_llama3: bool, max_gen_len: int=256, temperature: float = 0.8, top_p: float = 0.95):
    generator = load(ckpt_dir, tokenizer_path)
    prompts = ["The capital of Germany is the city of", "Here is my sonnet in the style of Shakespeare about an artificial intelligence:"]
    results = generator.generate_from_str(prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)

    for result in results:
        print(result)
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
