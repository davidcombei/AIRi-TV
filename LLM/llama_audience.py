from vllm import LLM, SamplingParams
from pypdf import PdfReader
import argparse
import os


def extract_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())


# 🔒 ADAUGARE MINIMA – GARANTEAZA #ANCHOR == #AUTHOR
def enforce_equal_turns(text: str) -> str:
    lines = [l for l in text.splitlines() if l.strip()]

    anchor_lines = [l for l in lines if l.startswith("[ANCHOR]")]
    author_lines = [l for l in lines if l.startswith("[AUTHOR]")]

    n = min(len(anchor_lines), len(author_lines))

    fixed_lines = []
    a_i = u_i = 0

    for line in lines:
        if line.startswith("[ANCHOR]") and a_i < n:
            fixed_lines.append(line)
            a_i += 1
        elif line.startswith("[AUTHOR]") and u_i < n:
            fixed_lines.append(line)
            u_i += 1

        if a_i == n and u_i == n:
            break

    return "\n".join(fixed_lines)


parser = argparse.ArgumentParser(description="Generate podcast conversation from PDF")
parser.add_argument("article_path", type=str, help="Path to the article PDF")
parser.add_argument("converstation_output", type=str, help="Directory where conversation.txt will be saved")
parser.add_argument("audience", type=str, help="Target audience of the conversation")


args = parser.parse_args()
pdf_path = args.article_path
output_dir = args.converstation_output
audience = args.audience


article_text = extract_pdf(pdf_path)




if audience == "liceeni":
    prompt = f"""
Generate a conversation between an anchor and an author based on the article below.

The conversation should be tailored for high school seniors (12th grade students), using simple and natural language.

FOCUS ONLY ON:
- university programs
- jobs/careers
- skills

Use ONLY information from the article. Do not invent facts.
Ignore technical details, research, and theory.

FORMAT:
[ANCHOR]: ...
[AUTHOR]: ...
One utterance per turn.

IMPORTANT:
Start exactly like this:
[ANCHOR]: Hey everyone! Not sure what to do after high school? Let’s talk about it. Today we have [Author's Name] with us.
[AUTHOR]: Thanks for having me!

Continue in the same format.

CONSTRAINTS:
- ONLY 12–14 turns total
- Equal turns for ANCHOR and AUTHOR
- Each reply 1–2 short sentences
- No repetition

DO NOT GENERATE ANYTHING ELSE. ONLY THE CONVERSATION.

ARTICLE:

{article_text}

Begin conversation:
"""

if audience == "politicieni":
    prompt = f"""
Generate a conversation between an anchor and an author based on the article below.

The conversation should be tailored for policymakers and politicians, using clear, formal, and strategic language.

FOCUS ONLY ON:
- public policy implications
- economic impact
- social impact
- governance and decision-making

Use ONLY information from the article. Do not invent facts.
Ignore technical details, deep theory, or irrelevant explanations.

FORMAT:
[ANCHOR]: ...
[AUTHOR]: ...
One utterance per turn.

IMPORTANT:
Start exactly like this:
[ANCHOR]: Welcome. Today we discuss key issues that matter for decision-makers. Joining us is [Author's Name].
[AUTHOR]: Thank you for the invitation.

Continue in the same format.

CONSTRAINTS:
- ONLY 12–14 turns total
- Equal turns for ANCHOR and AUTHOR
- Each reply 1–2 short sentences
- No repetition

DO NOT GENERATE ANYTHING ELSE. ONLY THE CONVERSATION.

ARTICLE:

{article_text}

Begin conversation:
"""


if audience == "firme":
    prompt = f"""
Generate a conversation between an anchor and an author based on the article below.

The conversation should be tailored for representatives of companies in the computer science and tech industry, using clear, professional, and business-oriented language.

FOCUS ONLY ON:
- market trends
- business opportunities in the market
- workforce and hiring needs
- required skill sets for employees

Use ONLY information from the article. Do not invent facts.
Ignore technical details, deep theory, or irrelevant explanations.

FORMAT:
[ANCHOR]: ...
[AUTHOR]: ...
One utterance per turn.

IMPORTANT:
Start exactly like this:
[ANCHOR]: Welcome. Today we discuss key insights relevant for the tech industry. Joining us is [Author's Name].
[AUTHOR]: Thank you for the invitation.

Continue in the same format.

CONSTRAINTS:
- ONLY 12–14 turns total
- Equal turns for ANCHOR and AUTHOR
- Each reply 1–2 short sentences
- No repetition

DO NOT GENERATE ANYTHING ELSE. ONLY THE CONVERSATION.

ARTICLE:

{article_text}

Begin conversation:
"""



llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    tensor_parallel_size=1,
)

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1500,
)

resp = llm.generate(prompt, params)

generated_text = resp[0].outputs[0].text

# 🔒 AICI SE GARANTEAZA EGALITATEA
generated_text = enforce_equal_turns(generated_text)

print(generated_text)

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "conversation.txt"), "w", encoding="utf-8") as f:
    f.write(generated_text)

