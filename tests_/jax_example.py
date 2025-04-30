import jax
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze
from jax_llama import convert_llama_weights, LLaMA, FlaxLLaMAForCausalLM, LLaMA3Tokenizer
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
import fire
def load(ckpt_dir: str, tokenizer_path: str, max_seq_length: int=2048, **model_kwargs) -> LLaMA:
    # setup jax mesh
    pass
"""
    devices = mesh_utils.create_device_mesh((1, len(jax.devices())))
    mesh = Mesh(devices, axis_names=('dp', 'mp'))
    print(f"Mesh: {mesh}")
    
    # load jax model
    tokenizer = LLaMA3Tokenizer(tokenizer_path)

    jax_params, jax_config = convert_llama_weights(ckpt_dir, tokenizer, max_seq_len=max_seq_length)
    with jax.default_device(jax.devices('cpu')[0]):
        jax_params = freeze(jax.tree_util.tree_map(lambda x: jnp.asarray(x), jax_params))
    # shard params
    param_spec = freeze(get_llama_param_partition_spec(unfreeze(jax_params), fsdp=False))
    jax_params = jax.tree_util.tree_map(lambda param, spec: jax.device_put(param, NamedSharding(mesh, spec)), jax_params, param_spec)

    # build model
    jax_model = FlaxLLaMAForCausalLM(jax_config, _do_init=False, **model_kwargs)

    return LLaMA(jax_params, jax_model, tokenizer, mesh=mesh)
"""