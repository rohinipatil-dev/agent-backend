# Autothinker Agent Backend

This is the backend for **Autothinker AI Agent Builder**, a no-code platform that generates and deploys AI-powered agents based on user prompts.

---

## ⚠️ Proprietary Notice

© 2025 AutoThinker AI. All rights reserved.

This repository contains proprietary and confidential code intended solely for evaluation purposes in connection with potential investment, partnership, or internal review.

No part of this codebase may be reproduced, distributed, or transmitted in any form or by any means without the prior written permission of AutoThinker AI.

---

## Features

- FastAPI backend server
- Uses GPT-4 to generate deployable Python apps
- Automatically creates `requirements.txt`
- Prepares code for deployment (Render, etc.)
- Designed for integration with the Autothinker frontend

---

## Installation

Clone this repository (access required):

```bash
git clone https://github.com/rohinipatil-dev/agent-backend.git
cd agent-backend

Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Set your OpenAI API Key:
export OPENAI_API_KEY="sk-..."

## Export .env
set -a
source .env
set +a

Run the server:
uvicorn main:app --reload --port 10000

## Deployment

This backend can be deployed to Render, Heroku, or any cloud platform that supports Python.

Recommended start command to generate the agent:
curl http://localhost:10000/build-agent      -H "Content-Type: application/json"      -d '{"prompt":"Create a chatbot that answers python programming questions"}'

Make sure to configure environment variables (such as OPENAI_API_KEY) in your deployment settings.

## Frontend
The backend is designed to integrate with the Autothinker Agent Forge frontend:

https://autothinker.org/

Frontend sends prompt data to the backend endpoint (/build-agent) and displays the generated app URL to the user.

## Roadmap
Automated GitHub repository creation for each agent

Render API integration for programmatic deployment

Usage metrics and analytics

LangChain support for advanced workflows

## Contact
For access requests, inquiries, or partnership discussions, please contact:

Rohini Patil
AutoThinker AI
rohini.patil@autothinker.org
