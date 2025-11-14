
from vllm import LLM, SamplingParams

model_name = "meta-llama/Meta-Llama-3.1-70B"

llm = LLM(
    model=model_name,
    dtype="bfloat16",
    tensor_parallel_size=2  # nr of gpus
)

prompt = "Hey, how are you doing today?"
params = SamplingParams(max_tokens=200)

output = llm.generate(prompt, params)
print(output[0].outputs[0].text)