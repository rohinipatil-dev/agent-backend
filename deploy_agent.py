import os
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a code-generating assistant. 
Your task is to build a deployable Python script for an AI agent based on the user prompt.

You must:
1. Use Streamlit for UI if the agent is interactive.
2. Use OpenAI API for conversation.
3. Integrate tools or APIs mentioned.
4. Keep code modular and clean.

Return only the code, no explanation or text.
"""


def extract_code(generated_text: str) -> str:
    text = generated_text.strip()

    code_block_match = re.search(r"```(?:python)?\n([\s\S]*?)```", text, re.IGNORECASE)
    if code_block_match:
        return code_block_match.group(1).strip()

    if (text.startswith("'''") and text.endswith("'''")) or (text.startswith('"""') and text.endswith('"""')):
        return text[3:-3].strip()

    lines = text.splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(("import ", "def ", "class ", "from ", "async ", "print(", "st.", "if ", "for ", "while ")):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines).strip()

    return text


def deploy_agent_code(prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.3,
        max_tokens=2000
    )

    generated_code_raw = response.choices[0].message.content
    generated_code = extract_code(generated_code_raw)

    repo_path = "generated_agent"
    os.makedirs(repo_path, exist_ok=True)

    with open(os.path.join(repo_path, "app.py"), "w") as f:
        f.write(generated_code)

    requirements = ["streamlit", "openai"]
    lowered_code = generated_code.lower()
    if "gensim" in lowered_code:
        requirements.append("gensim")
    if "youtube_transcript_api" in lowered_code:
        requirements.append("youtube_transcript_api")
    if "transformers" in lowered_code:
        requirements.append("transformers")
    if "requests" in lowered_code:
        requirements.append("requests")

    requirements = list(set(requirements))

    with open(os.path.join(repo_path, "requirements.txt"), "w") as f:
        f.write("\n".join(requirements))

    return "https://your-agent-app.onrender.com"
