from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deploy_agent import deploy_agent_code

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

@app.post("/build-agent")
async def build_agent(prompt: Prompt):
    try:
        deployment_url = deploy_agent_code(prompt.prompt)
        return {"url": deployment_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

