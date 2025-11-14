from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)

article = open("articol.txt").read()

messages = [
    {
        "role": "system",
        "content": "You are a system for generating news conversations."
    },
    {
        "role": "user",
        "content": f"""
Generate a conversation between an ANCHOR and an AUTHOR based on the article below.
They speak one utterance at a time.

Format:
[ANCHOR]: text
[AUTHOR]: text

ARTICLE:
<<<ARTICLE_START>>>
{article}
<<<ARTICLE_END>>>
"""
    }
]

input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.8,
    top_p=0.95
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
