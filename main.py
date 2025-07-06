from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deploy_agent import deploy_agent

app = FastAPI()

# Add this block to enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    prompt: str

@app.post("/build-agent")
async def build_agent(prompt: Prompt):
    try:
        deployment_url = deploy_agent(prompt.prompt)
        return {"url": deployment_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

