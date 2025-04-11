from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model_utils import get_answer
import time
import wandb
import asyncio

# W&B Setup
wandb.init(project="qa-model-deployment", name="fastapi-inference")

app = FastAPI(title="Question Answering API")

class QARequest(BaseModel):
    context: str = Field(..., example="The sky is blue because of the way sunlight interacts with Earth's atmosphere.")
    question: str = Field(..., example="Why is the sky blue?")


class QAResponse(BaseModel):
    answer: str
    score: float
    latency_ms: float

@app.on_event("shutdown")
def shutdown_event():
    wandb.finish()

@app.post("/predict", response_model=QAResponse)
async def predict(data: QARequest):
    try:
        start = time.perf_counter()
        result = await asyncio.to_thread(get_answer, data.context, data.question)
        latency = (time.perf_counter() - start) * 1000  # ms

        wandb.log({
            "inference_latency_ms": latency,
            "score": result.get("score", -1),
            "question": data.question,
            "context_length": len(data.context)
        })

        return QAResponse(
            answer=result.get("answer", ""),
            score=result.get("score", 0.0),
            latency_ms=latency
        )
    except Exception as e:
        wandb.log({"error": str(e)})
        raise HTTPException(status_code=500, detail="Internal model error")
