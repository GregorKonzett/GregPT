from contextlib import asynccontextmanager

from fastapi import FastAPI

from inference.InferenceService import InferenceService
from inference.model.models import InferenceRequest, InferenceResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_service = InferenceService()
    app.state.inference_service.start()
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def get_inference(req: InferenceRequest) -> InferenceResponse:
    print(f"Sending {req.prompt} to the LLM")

    res = await app.state.inference_service.enqueue(req)
    return InferenceResponse(response=res)
