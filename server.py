import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from agent import run_agent

app = FastAPI()

HTML_PATH = Path(__file__).parent / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PATH.read_text(encoding="utf-8")


class RunRequest(BaseModel):
    objective: str


@app.post("/run")
async def run(req: RunRequest):
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_event(event: dict):
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def run_in_thread():
        try:
            run_agent(req.objective, on_event=on_event)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    async def event_stream():
        loop.run_in_executor(None, run_in_thread)
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
