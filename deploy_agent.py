import os
import re
import sys
import ast
import time
import uuid
import logging
import pkgutil
from datetime import datetime
from threading import Lock
from github import Github
import requests
from openai import OpenAI
import json


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lock to handle concurrency so only one deployment can happen at a time
deploy_lock = Lock()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an expert software engineer.
Your task is to build a deployable Python script for an AI agent based on the user prompt.
Generate code that uses only supported classes and methods from the latest stable version of the specified language or library.

Requirements:
1. Use Streamlit for UI if the agent is interactive.
2. Use ONLY the OpenAI Python SDK version >=1.0.0.
3. Do NOT use or import `openai.Completion`, `openai.ChatCompletion`, or any deprecated models like `text-davinci-003` or `text-davinci-002`.
4. For all OpenAI completions, use chat models: ONLY `gpt-3.5-turbo` or `gpt-4` with `client.chat.completions.create(...)`.
5. Import ONLY: `from openai import OpenAI`
6. Create the client using: `client = OpenAI()`
7. For chat completions, use:
   response = client.chat.completions.create(
       model="gpt-5",  # or "gpt-4"
       messages=[{"role": "system", "content": "You are a helpful assistant."}, ...]
   )
8. Access the response text using: `response.choices[0].message.content`
9. Do NOT include any API key in the code.
10. The generated script must be a valid Streamlit app that can run standalone.
11. Keep the code modular and clean.

Return only the Python code, no explanation or formatting. Output must be valid code that can be saved as `app.py` and run with Streamlit.

Any code using deprecated OpenAI APIs or models will be rejected.
"""

# Extracts Python code from the generated response
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

# Ensures OpenAI SDK usage meets expected guidelines
def validate_openai_api_usage(code: str):
    deprecated_patterns = [
        "openai.Completion", "openai.ChatCompletion", "openai.ChatCompletion.create",
        "openai.api_key", "engine="
    ]
    for pattern in deprecated_patterns:
        if pattern in code:
            raise ValueError(f"Deprecated OpenAI API usage found: {pattern}")
    return

# Generates a requirements.txt from the generated code
def build_requirements_txt(code: str) -> str:
    import sys

    # Parse the code into an AST
    tree = ast.parse(code)

    # Collect all imported module names
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split('.')[0])

    # These are definitely standard regardless of Python version
    built_in_stdlib = {
        "os", "sys", "re", "math", "time", "datetime", "json", "typing", "random",
        "pathlib", "logging", "collections", "subprocess", "threading", "itertools",
        "functools", "http", "urllib", "shutil", "queue", "traceback", "enum", "base64", "io"
    }

    # Get standard library module names using sys.stdlib_module_names if available (Python 3.10+)
    if hasattr(sys, "stdlib_module_names"):
        stdlib_modules = set(sys.stdlib_module_names)
    else:
        stdlib_modules = built_in_stdlib  # fallback for older Python versions

    # Final list of external dependencies
    external_packages = sorted(mod for mod in imported_modules if mod not in stdlib_modules and mod not in built_in_stdlib)

    # Always include streamlit and openai
    external_packages.extend(["streamlit", "openai"])

    return "\n".join(sorted(set(external_packages)))

# Main function to deploy an agent based on user prompt
def deploy_agent(prompt: str) -> str:
    with deploy_lock:
        logger.info("Starting deployment for new agent prompt")

        # Construct conversation for OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        logger.info("Calling OpenAI API to generate code")
        response = client.chat.completions.create(
            model="gpt-5",
            messages=messages
        )

        # Extract the generated Python code
        generated_code_raw = response.choices[0].message.content
        # Log the raw output so you can inspect exactly what the model returned
        logger.debug("RAW MODEL OUTPUT:\n%s", generated_code_raw)

        generated_code = extract_code(generated_code_raw)
        # Also log the extracted code
        logger.debug("EXTRACTED CODE:\n%s", generated_code if generated_code else "[EMPTY]")

        # Validate and build requirements
        validate_openai_api_usage(generated_code)
        requirements_txt = build_requirements_txt(generated_code)

        # Get credentials from environment
        github_token = os.environ["GITHUB_TOKEN"]
        github_username = os.environ["GITHUB_USERNAME"]
        render_api_key = os.environ["RENDER_API_KEY"]
        openai_api_key = os.environ["OPENAI_API_KEY"]
        render_owner_id = os.environ["RENDER_OWNER_ID"]

        # Create a unique GitHub repository
        logger.info("Creating unique GitHub repo")
        g = Github(github_token)
        user = g.get_user()

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        repo_name = f"agent-{timestamp}-{unique_id}"

        repo = user.create_repo(name=repo_name, private=False, auto_init=True)

        # Upload files to the repo
        logger.info(f"Repo created: {repo_name}. Uploading files")
        repo.create_file("app.py", "Add app.py", content=generated_code, branch="main")
        repo.create_file("requirements.txt", "Add requirements.txt", content=requirements_txt, branch="main")

        # Prepare deployment on Render
        render_api_url = "https://api.render.com/v1/services"
        headers = {
            "Authorization": f"Bearer {render_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "ownerId": render_owner_id,  # Replace with your actual ownerId
            "name": repo_name,
            "repo": f"https://github.com/{github_username}/{repo_name}",
            "branch": "main",
            "type": "web_service",
            "plan": "starter",
            "envVars": [
                {"key": "OPENAI_API_KEY", "value": openai_api_key}
                ],
            "serviceDetails": {
                "env": "python",
                "envSpecificDetails": {
                    "buildCommand": "pip install -r requirements.txt",
                    "startCommand": "streamlit run app.py",
                    "pythonVersion": "3.10"
                },
            "autoDeploy": True
            }
        }

        #logger.info(f"Payload to Render: {json.dumps(payload, indent=2)}")
        logger.info("Triggering Render deployment")

        render_response = requests.post(render_api_url, headers=headers, json=payload)
        if not render_response.ok:
            #logger.error(f"Failed payload: {json.dumps(payload)}")
            logger.error(f"Response: {render_response.text}")
            raise Exception(f"Render API call failed: {render_response.text}")

        # Wait briefly for service registration
        time.sleep(3)

        # Get service ID from the response
        service_data = render_response.json()
        service_id = service_data.get("service", {}).get("id")
        if not service_id:
            raise Exception(f"No service ID returned from Render")

        logger.info(f"Deployment started for service ID: {service_id}")

        # Poll for deployment status
        max_wait_time = 300  # seconds
        poll_interval = 10  # seconds
        elapsed_time = 0
        deployment_url = None

        while elapsed_time < max_wait_time:
            deploys_resp = requests.get(
                f"https://api.render.com/v1/services/{service_id}/deploys",
                headers=headers
            )
            if deploys_resp.ok:
                deploys = deploys_resp.json()
                if deploys and isinstance(deploys, list):
                    latest = deploys[0]
                    status = latest.get("status")
                    logger.info(f"Deployment status: {status}")
                    if status == "live":
                        deployment_url = service_data.get("service", {}).get("serviceDetails", {}).get("url")
                        break
                    elif status in ["build_failed", "update_failed", "canceled"]:
                        raise Exception(f"Deployment failed: status={status}")
            time.sleep(poll_interval)
            elapsed_time += poll_interval

        # If no URL from serviceDetails, construct default URL
        if not deployment_url:
            deployment_url = f"https://{repo_name}.onrender.com"

        logger.info(f"Deployment completed. App URL: {deployment_url}")
        return deployment_url
