from generation import Llama
import os
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist  # type: ignore
# Only initialize once
if not dist.is_initialized():
    os.environ.setdefault("RANK", "0")
    
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    fs_init.initialize_model_parallel(1)
    
def run():
    ckpt_dir = "/root/tt/3_1_8b/Llama-Jax-Paralelism/llama3.1-8B/8B"
    tokenizer_path = "/root/tt/3_1_8b/Llama-Jax-Paralelism/llama3.1-8B/original/tokenizer.model"
    prompt = (
        "Q: A bumper car rink has 12 red cars. They have 2 fewer green cars than they have red cars. "
        "They have 3 times the number of blue cars as they have green cars. The rink also has yellow cars. "
        "If the rink has 75 cars in total how many yellow cars do they have?\n"
        "A:"
    )

    max_seq_len = 256
    max_batch_size = 1
    max_gen_len = 64

    # Initialize Llama model and tokenizer
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=1,  
    )

    # Generate answer
    results = llama.text_completion(
        prompts=[prompt],
        temperature=0,  # greedy decoding
        top_p=1.0,
        max_gen_len=max_gen_len,
        echo=False,
        logprobs=False,
    )

    print("\n🧠 Output:")
    print(results[0]["generation"])

if __name__ == "__main__":
    run()
