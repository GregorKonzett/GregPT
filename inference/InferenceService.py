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
            res = await asyncio.to_thread(self.infer, job.prompt)

            if not job.future.cancelled():
                job.future.set_result(res)

    def infer(self, prompt: str) -> str:
        encoded_query = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=get_device())
        out = self.gpt.generate(encoded_query)
        out = self.tokenizer.decode(out[0].cpu().tolist()).strip()
        return out


    async def enqueue(self, req: InferenceRequest):
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await self.inference_queue.put(InferenceJob(prompt=req.prompt, future=future))

        return await future


