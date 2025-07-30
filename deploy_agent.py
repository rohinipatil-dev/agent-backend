import os
import re
from datetime import datetime
from github import Github
import requests
from openai import OpenAI

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a code-generating assistant.
Your task is to build a deployable Python script for an AI agent based on the user prompt.

Requirements:
1. Use Streamlit for UI if the agent is interactive.
2. Use the OpenAI Python SDK version >=1.0.0 only.
3. Do NOT import or use `openai.Completion`, `openai.ChatCompletion`, or `openai.ChatCompletion.create`.
   - Do NOT import them under any circumstances.
   - Do NOT reference them in code.
4. Import ONLY: `from openai import OpenAI`
5. Create the client using: `client = OpenAI()`
6. For all completions, use: `response = client.chat.completions.create(...)`
7. Access the response text using: `response.choices[0].message.content`
8. Do NOT include any API key in the code.
9. The generated script must be a valid Streamlit app that can run standalone.
10. Keep the code modular and clean.

Return only the Python code, no explanation or formatting. Output must be valid code that can be saved as `app.py` and run with Streamlit.

Any code using deprecated OpenAI APIs will be rejected.
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
        if line.strip().startswith((
            "import ", "def ", "class ", "from ", "async ", "print(", "st.", "if ", "for ", "while "
        )):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines).strip()

    return text

def validate_openai_api_usage(code: str):
    deprecated_patterns = [
        "openai.Completion", "openai.ChatCompletion", "openai.ChatCompletion.create",
        "openai.api_key", "engine="
    ]
    for pattern in deprecated_patterns:
        if pattern in code:
            raise ValueError(f"Deprecated OpenAI API usage found: {pattern}")
    return

def deploy_agent(prompt: str) -> str:
    # Generate code
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

    # Validate the generated_code
    validate_openai_api_usage(generated_code)

    # Build requirements
    requirements = ["streamlit", "openai"]
    lowered = generated_code.lower()
    if "gensim" in lowered:
        requirements.append("gensim")
    if "youtube_transcript_api" in lowered:
        requirements.append("youtube_transcript_api")
    if "transformers" in lowered:
        requirements.append("transformers")
    if "requests" in lowered:
        requirements.append("requests")

    requirements_txt = "\n".join(sorted(set(requirements)))

    # GitHub and Render credentials
    github_token = os.environ["GITHUB_TOKEN"]
    github_username = os.environ["GITHUB_USERNAME"]
    render_deploy_hook_url = os.environ["RENDER_DEPLOY_HOOK_URL"]

    # Initialize GitHub
    g = Github(github_token)

    # Get the existing repo (IMPORTANT: this must be the repo your Render service uses)
    repo = g.get_user().get_repo("agent-template")

    # Delete old app.py if exists
    try:
        contents = repo.get_contents("app.py")
        repo.delete_file(
            contents.path,
            "Delete old app.py",
            contents.sha,
            branch="main"
        )
    except Exception:
        pass

    # Create new app.py
    repo.create_file(
        "app.py",
        "Add new app.py",
        content=generated_code,
        branch="main"
    )

    # Delete old requirements.txt if exists
    try:
        contents = repo.get_contents("requirements.txt")
        repo.delete_file(
            contents.path,
            "Delete old requirements.txt",
            contents.sha,
            branch="main"
        )
    except Exception:
        pass

    # Create new requirements.txt
    repo.create_file(
        "requirements.txt",
        "Add new requirements.txt",
        content=requirements_txt,
        branch="main"
    )

    # Trigger redeploy
    response = requests.post(render_deploy_hook_url)
    if not response.ok:
        raise Exception(f"Render deploy hook failed: {response.text}")

    # Return the URL of your agent-template Render app
    return "https://agent-template-e5fi.onrender.com"

