import argparse
import asyncio
import datetime
import json
import os
import random
import docker
from dataclasses import dataclass, field
from itertools import count
from typing import AsyncGenerator
import time

import aiohttp
import numpy as np
import pynvml
from codecarbon import OfflineEmissionsTracker
from tqdm.asyncio import tqdm

from benchmark.dataset import Dataset
from benchmark.runs.run import Run


SYSTEM_PROMPT = "You are an artificial intelligence assistant that gives helpful answers to the user's questions or instructions."
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=3 * 3600)
STOP_SEQUENCES = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]

@dataclass
class ResultIntermediate:
    task_id: int = field(default_factory=count().__next__)
    success: bool = True
    latency: float = 0.0
    prompt: str = ""
    response_bytes: list[bytes] = field(default_factory=list)

@dataclass
class Result:
    task_id: str = ""
    success: bool = True
    latency: float = 0.0
    prompt: str = ""
    response: str = ""
    num_prompt_tokens: int = 0
    num_completion_tokens: int = 0
    energy: float = 0.0

@dataclass
class Results:
    model: str
    backend: str
    gpu_model: str
    max_num_seqs: int
    request_rate: float
    num_requests: int
    num_failures: int = 0
    total_runtime: float = 0.0
    requests_per_second: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    latency_per_request: float = 0.0
    latency_per_output_token: float = 0.0
    server_side_total_energy: float = 0.0
    server_side_energy_per_request: float = 0.0
    server_side_energy_per_output_token: float = 0.0
    server_side_average_power: float = 0.0
    client_side_total_energy: float = 0.0
    client_side_energy_per_request: float = 0.0
    client_side_energy_per_output_token: float = 0.0
    client_side_average_power: float = 0.0
    results: list[Result] = field(default_factory=list)

def strip_stop_sequence(text: str, stop_sequences: list[str]) -> str:
    for stop in stop_sequences:
        if text.endswith(stop):
            return text[:-len(stop)]
    return text

async def send_request(
        result_intermediate: ResultIntermediate,
        model: str,
        api_url: str,
        prompt: str,
) -> None:
    headers = {"Content-Type": "application/json"}
    # OpenAI Chat Completions API request format
    # Assuming `add_generation_prompt` is either not needed or set to true
    pload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.0,
        "stop": STOP_SEQUENCES,
    }

    async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT) as session:
        request_start_time = time.perf_counter()
        async with session.post(api_url, headers=headers, json=pload) as response:
            # Request failed
            if response.status >= 300:
                print(f"Request failed: {await response.text()}")
                result_intermediate.prompt = prompt
                result_intermediate.success = False
                return
            chunks = []
            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
            request_end_time = time.perf_counter()

    result_intermediate.latency = request_end_time - request_start_time
    result_intermediate.prompt = prompt
    result_intermediate.response_bytes = chunks


async def get_request(input_requests: list[str],
                      result_intermediates: list[ResultIntermediate],
                      request_rate: float,
                      ) -> AsyncGenerator[tuple[ResultIntermediate, str], None]:
    if request_rate == float("inf"):
        # If the request rate is infinity, then we don't need to wait.
        for item in zip(result_intermediates, input_requests, strict=True):
            yield item
        return

    for item in zip(result_intermediates, input_requests, strict=True):
        yield item

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

async def benchmark(
    results: Results,
    model: str,
    api_url: str,
    input_requests: list[str],
    request_rate: float,
) -> None:
    tasks: list[asyncio.Task] = []
    result_intermediates = [ResultIntermediate() for _ in input_requests]
    pbar = tqdm(total=len(input_requests))
    async for ri, prompt in get_request(input_requests, result_intermediates, request_rate):
        pbar.update(1)
        task = asyncio.create_task(
            # Ensures results has same ordering as the input dataset
            send_request(ri, model, api_url, prompt)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)

    for result, intermediate in zip(results.results, result_intermediates, strict=True):
        result.task_id = intermediate.task_id
        result.success = intermediate.success
        result.latency = intermediate.latency
        result.prompt = intermediate.prompt
        if result.success:
            output = json.loads(b"".join(intermediate.response_bytes).decode("utf-8"))
            result.response = strip_stop_sequence(output["choices"][0]["text"], STOP_SEQUENCES)
            result.num_prompt_tokens = output["usage"]["prompt_tokens"]
            result.num_completion_tokens = output["usage"]["completion_tokens"]

def start_container(client: docker.DockerClient, model: str, port: int):
    try:
        container =  client.containers.get(f"vLLM-{model.split('/')[1]}")
        print(container.status)
        if container.status == "running":
            print("container already running")
            return True
    except: 
        print("no container of model found")

    params = f"--model {model}"
    container = client.containers.run(
        "vllm/vllm-openai:latest", 
        command=params,
        name=f"vLLM-{model.split('/')[1]}", runtime="nvidia", 
        ports={8000: 8888}, ipc_mode="host", detach=True
        )
    print("starting container")
    
    timeout = 120
    pauze = 4
    elapsed_time = 0
    while container.status != 'RUNNING' and elapsed_time < timeout:
        time.sleep(pauze)
        elapsed_time += pauze
        print(f"elapsed time: {elapsed_time} - Status: {container.status}")
        continue
    return True

def stop_container(client: docker.DockerClient, model: str):
    for container in client.containers.list():
        if container.name != f"vLLM-{model.split('/')[1]}":
            container.stop()

class vLLMRun(Run):
    def __init__(self, model: str, dataset: Dataset, passes: int, request_rate: float = float('inf')):
        super().__init__(model, dataset, passes)
        self.name = f"vLLM-{model.split('/')[1]}"
        self.api_url = "http://localhost:8888/v1/completions"
        self.request_rate: float = request_rate
        self.client = docker.from_env()

    async def start(self):
        stop_container(self.client, model=self.model_name)
        start_container(self.client, model=self.model_name, port=8888)
        input_requests = self.dataset.get_list(0, self.passes)

        tracker = OfflineEmissionsTracker(
            log_level="warning",
            tracking_mode="machine",
            allow_multiple_runs=True,
            output_file="codecarbon.csv",
            country_iso_code="NLD",
            experiment_id=f"{self.passes}passes-{self.model_name}-{datetime.UTC}"
        )
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_model = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()

        results = Results(
            model=self.model_name,
            backend="vLLM",
            gpu_model=gpu_model,
            max_num_seqs=100,
            request_rate=self.request_rate,
            num_requests=len(input_requests),
            results=[Result() for _ in input_requests],
        )

        tracker.start()
        await benchmark(results, self.model_name, self.api_url, input_requests, self.request_rate)
        tracker.stop()

        self.emissions_data = tracker.final_emissions_data

