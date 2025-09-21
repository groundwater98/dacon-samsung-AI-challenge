import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# -------------------------------
# Parse external arguments
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
parser.add_argument("--max_seqs", type=int, default=128, help="Maximum number of sequences")
args = parser.parse_args()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# Load vLLM model
llm = LLM(
    model=args.model_path,
    dtype="float16",
    tensor_parallel_size=args.tp_size,
    gpu_memory_utilization=0.9,
    max_num_seqs=args.max_seqs
)

# Default sampling parameters
default_sampling = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024
)

print("CLI Inference (Thinking Mode ON, type 'exit' to quit)\n")
while True:
    user_prompt = input("Prompt >>> ")
    if user_prompt.lower() in ["exit", "quit"]:
        break

    # Always use thinking mode template
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    # Generate with vLLM
    outputs = llm.generate([text], default_sampling)
    generated_text = outputs[0].outputs[0].text

    print("\n[Raw Model Output]")
    print(generated_text)

    # Separate thinking block if present
    if "</think>" in generated_text:
        thinking, answer = generated_text.split("</think>", 1)
        print("\n[Thinking]")
        print(thinking.strip())
        print("\n[Answer]")
        print(answer.strip())
    else:
        print("\n[Answer]")
        print(generated_text.strip())

    print("-" * 60)

