import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.sse import EventSourceResponse

from inference.InferenceService import InferenceService
from inference.model.models import InferenceRequest, InferenceResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_service = InferenceService()
    app.state.inference_service.start()
    yield

app = FastAPI(lifespan=lifespan)


@app.post("/inference", response_class=EventSourceResponse)
async def infer(req: InferenceRequest) -> AsyncGenerator[InferenceResponse, None]:
    resp_queue: asyncio.Queue = await app.state.inference_service.enqueue(req)

    while True:
        curr_val = await resp_queue.get()

        if curr_val is None:
            break

        yield InferenceResponse(response=curr_val)
