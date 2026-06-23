import asyncio

import torch

from inference.model.models import InferenceJob, InferenceRequest
from model.GptModel import GptModel, get_device
from tokenizer.TikTokenTokenizer import TikTokenTokenizer
from weights.GCPStorageWeightLoader import GCPStorageWeightLoader


class InferenceService:
    def __init__(self):
        self.tokenizer = TikTokenTokenizer()
        self.gpt = GptModel(vocab_size=self.tokenizer.get_vocab_size(), tokenizer=self.tokenizer).to(get_device())

        weight_loader = GCPStorageWeightLoader()
        weight_loader.load_checkpoint(self.gpt, False)

        self.inference_queue: asyncio.Queue[InferenceJob] = asyncio.Queue()

    def start(self):
        asyncio.create_task(self._worker_loop())

    async def _worker_loop(self):
        while True:
            job = await self.inference_queue.get()
            await asyncio.to_thread(self.infer, job.prompt, job.resp_queue)
            await job.resp_queue.put(None)

    def infer(self, prompt: str, resp_queue: asyncio.Queue):
        encoded_query = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=get_device())
        asyncio.run(self.gpt.async_generate(encoded_query, resp_queue=resp_queue))


    async def enqueue(self, req: InferenceRequest):
        resp_queue = asyncio.Queue()

        await self.inference_queue.put(InferenceJob(prompt=req.prompt, resp_queue=resp_queue))

        return resp_queue


