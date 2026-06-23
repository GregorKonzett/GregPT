import asyncio

from attr import dataclass
from pydantic import BaseModel


class InferenceResponse(BaseModel):
    response: str

class InferenceRequest(BaseModel):
    prompt: str

@dataclass
class InferenceJob:
    prompt: str
    resp_queue: asyncio.Queue