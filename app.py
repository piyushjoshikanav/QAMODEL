from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import get_answer
import time
import wandb

# Initialize W&B once when app starts
wandb.init(project="qa-model-deployment", name="fastapi-inference")

app = FastAPI(title="Question Answering API")

class QARequest(BaseModel):
    context: str
    question: str

class QAResponse(BaseModel):
    answer: str
    score: float
    latency_ms: float  # new field for latency

@app.post("/predict", response_model=QAResponse)
def predict(data: QARequest):
    start = time.perf_counter()
    result = get_answer(data.context, data.question)
    latency = (time.perf_counter() - start) * 1000  # in ms

    # Log metrics to W&B
    wandb.log({
        "inference_latency_ms": latency,
        "score": result["score"]
    })

    return QAResponse(answer=result["answer"], score=result["score"], latency_ms=latency)
