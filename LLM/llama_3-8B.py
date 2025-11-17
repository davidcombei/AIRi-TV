from vllm import LLM, SamplingParams
from pypdf import PdfReader
import argparse
import os
def extract_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)


parser = argparse.ArgumentParser(description="Generate podcast conversation from PDF")
parser.add_argument("article_path", type=str, help="Path to the article PDF")
parser.add_argument("converstation_output", type=str, help="Directory where conversation.txt will be saved")

args = parser.parse_args()
pdf_path = args.article_path
output_dir = args.converstation_output

article_text = extract_pdf(pdf_path)

prompt = f"""
Generate a conversation between an anchor and an author based on the article below.
The conversation should cover the main points of the article in a question-and-answer format.
Make it as long as possible, but keep it relevant to the article content and to not exceed 40-50 turns.
Each turn will start with the participant's role in square brackets, followed by a colon and their utterance. Make sure that all utterances have the speaker annotation.:
[ANCHOR]: ...
[AUTHOR]: ...
Only one utterance per turn. Use only information from the article.
Do not invent facts not found in the article.
Make the conversation engaging and informative.
Make it sound natural and human-like.
Ignore the Acknowledgment section of the article.
Ignore the links in the article.
Ignore the references in the article.
Try to discuss a little bit about the results presented in the tables of the articles.
Make the first part of the entry utterance using: Good day everyone, welcome to our show. Today we have with us [Author's Name], the author of the article. Thank you for joining us.
Make sure that the closing is natural, not leaving a question in the air.
Make sure that there are only replies from the ANCHOR and AUTHOR, no other speakers or sentences added.
VERY IMPORTANT: THE NR OF TURNS SHOULD BE EQUAL FOR BOTH ANCHOR AND AUTHOR.
DO NOT GENERATE ANYTHING ELSE. ONLY PROVIDE THE CONVERSATION WITH THE ANNOTATIONS.
Here is the
ARTICLE:

{article_text}


Begin conversation:
"""

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    tensor_parallel_size=1,
    #enable_torch_compile=False,
)

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1500,
)

resp = llm.generate(prompt, params)
generated_text = resp[0].outputs[0].text
print(resp[0].outputs[0].text)

with open(os.path.join(output_dir,"conversation.txt"), "w", encoding="utf-8") as f:
    f.write(generated_text)
