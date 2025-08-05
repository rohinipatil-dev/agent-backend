import os
import re
import sys
import ast
import pkgutil
import time
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
2. Use ONLY the OpenAI Python SDK version >=1.0.0.
3. DO NOT use or import:
   - openai.Completion
   - openai.ChatCompletion
   - Any deprecated models like text-davinci-003, text-davinci-002, or gpt-3.5-turbo-0301
   - Anything from openai.api_resources or its submodules
   - Specifically, DO NOT write or suggest: from openai.api_resources import Completion
   - DO NOT use or suggest client.text_embedding.create(...) or openai.Embedding.create(...) — it is invalid and will cause an AttributeError
4. For all OpenAI completions, use chat models ONLY: gpt-3.5-turbo or gpt-4 with client.chat.completions.create(...).
5. Import ONLY: from openai import OpenAI
6. Create the client using: client = OpenAI()
7. For chat completions, use:
   response = client.chat.completions.create(
       model="gpt-3.5-turbo",  # or "gpt-4"
       messages=[{"role": "system", "content": "You are a helpful assistant."}, ...]
   )
8. Access the response text using: response.choices[0].message.content
9. Do NOT include any API key in the code.
10. The generated script must be a valid Streamlit app that can run standalone.
11. Keep the code modular and clean.

If the prompt involves Retrieval-Augmented Generation (RAG):
12. Accept a PDF upload from the user.
13. Extract text from the PDF using: from PyPDF2 import PdfReader
14. Use PdfReader(uploaded_file), loop through pdf.pages, and call page.extract_text() to get content.
15. Chunk the text into approximately 500-character segments.
16. Generate embeddings for each chunk using text-embedding-3-small with the OpenAI SDK.
    Use:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )
    Access the embedding via: response.data[0].embedding
17. Use ChromaDB to store and search vector data. Use either in-memory or persistent DB:

```python
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="./chroma_db")  # or chromadb.Client() for in-memory
collection = client.get_or_create_collection(name="docs")

# To insert documents:
collection.add(
    documents=[chunk],  # or list of chunks
    embeddings=[embedding],  # or list of embeddings
    ids=[f"id_{i}"]  # unique ID per document
)

# To query:
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)```
18. For user queries:
    - Generate an embedding.
    - Use ChromaDB to retrieve top 3 relevant chunks.
    - Construct a context-based prompt from those chunks and query.
    - Use client.chat.completions.create(...) to generate the answer.

Library and import rules:
19. Required third-party libraries: streamlit, PyPDF2, chromadb, tiktoken, openai
20. Never use LangChain or any other external framework.
21. Never use deprecated PyPDF2 interfaces like PdfFileReader, getPage(), or any import from PyPDF2.pdf. Only use PdfReader and pdf.pages[i] with extract_text().
22. Never use or suggest faiss as a vector store.
23. Never mention or include from chromadb import ChromaDB.
24. Do not use any deprecated OpenAI, PyPDF2, or ChromaDB APIs or functions.

tiktoken rules:
25. NEVER write or suggest: from tiktoken import Tokenizer or from tiktoken import models or anything similar. These are invalid and will break the app.
26. Do NOT use or reference the Tokenizer class from tiktoken.
27. If token counting is needed, use:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
28. Do NOT import from openai.api_resources or any of its submodules such as openai.api_resources.experimental.

Final output:
29. Return ONLY the full Python code in a single block. Do NOT include explanations, comments, or markdown outside the code block.

Any code violating these instructions will be considered invalid.
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
    forbidden_patterns = [
        "openai.Completion",
        "openai.ChatCompletion",
        "openai.ChatCompletion.create",
        "openai.api_key",
        "engine=",
        "from tiktoken import Tokenizer",
        "from tiktoken import models",
        "Tokenizer(",
        "from openai.api_resources",
        "openai.api_resources.",
    ]
    for pattern in forbidden_patterns:
        if pattern in code:
            raise ValueError(f"Invalid or deprecated usage found: {pattern}")
    return


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
    requirements_txt = build_requirements_txt(generated_code)

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

    #Add a delay of 5 seconds before triggering redploy
    time.sleep(5)

    # Trigger redeploy
    response = requests.post(render_deploy_hook_url)
    if not response.ok:
        raise Exception(f"Render deploy hook failed: {response.text}")

    # Return the URL of your agent-template Render app
    return "https://agent-template-e5fi.onrender.com"

