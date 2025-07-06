from dotenv import load_dotenv
load_dotenv()

import os
import re
from datetime import datetime
import requests
from github import Github
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

def deploy_agent(prompt: str) -> str:
    # Generate code with GPT
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

    # Build requirements.txt dynamically
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
    requirements_txt = "\n".join(requirements)

    # GitHub and Render credentials
    github_token = os.environ["GITHUB_TOKEN"]
    github_username = os.environ["GITHUB_USERNAME"]
    render_deploy_hook_url = os.environ["RENDER_DEPLOY_HOOK_URL"]

    # Initialize GitHub
    g = Github(github_token)

    # Create unique repo name
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    repo_name = f"agent-{timestamp}"

    # Create the new repo
    repo = g.get_user().create_repo(
        name=repo_name,
        description="Auto-generated agent from AutoThinker AI",
        private=False,
        auto_init=False,
    )

    # Add app.py
    repo.create_file(
        "app.py",
        "Add app.py",
        content=generated_code,
        branch="main"
    )

    # Add requirements.txt
    repo.create_file(
        "requirements.txt",
        "Add requirements.txt",
        content=requirements_txt,
        branch="main"
    )

    # Trigger Render deploy
    response = requests.post(render_deploy_hook_url)
    if not response.ok:
        raise Exception(f"Render deploy hook failed: {response.text}")

    # Return the live Render URL
    # render_url = f"https://{repo_name}.onrender.com"

    # Return the existing Render service URL
    render_url = "https://agent-template.onrender.com"
    return render_url

