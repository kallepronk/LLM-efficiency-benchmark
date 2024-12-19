import datetime
import json
import typing

from tqdm.auto import tqdm

from codecarbon.output_methods.emissions_data import EmissionsData

from benchmark.runs.run import Run
import os


class Benchmark:
    def __init__(self, runs: [Run], name: str):
        self.runs = runs
        self.name = f"{name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def run(self):
        # Preform the tests
        pbar = tqdm(total=len(self.runs), desc="Total progress", position=0, leave=True)
        for run in self.runs:
            run.start()
            pbar.update()

    async def run_async(self):
        # Preform the tests
        pbar = tqdm(total=len(self.runs), desc="Total progress", position=0, leave=True)
        for run in self.runs:
            await run.start()
            pbar.update()


    def collect_results(self):
        for run in self.runs:
            if not run.has_finished():
                raise AssertionError(f"Run {run.name} has not finished")


        general_data: EmissionsData = self.runs[1].emissions_data
        total_duration: float = 0
        total_cpu_energy: float = 0
        total_gpu_energy: float = 0
        total_ram_energy: float = 0
        runs = []

        for run in self.runs[1:]:
            total_duration += run.emissions_data.duration
            total_cpu_energy += run.emissions_data.cpu_energy
            total_gpu_energy += run.emissions_data.gpu_energy
            total_ram_energy += run.emissions_data.ram_energy

            runs.append(
                {
                    "name": run.name,
                    "passes": run.passes,
                    "model": run.model_name,
                    "dataset": run.dataset.name,
                    "timestamp": run.emissions_data.timestamp,
                    "duration": run.emissions_data.duration,
                    "cpu_energy": run.emissions_data.cpu_energy,
                    "gpu_energy": run.emissions_data.gpu_energy,
                    "ram_energy": run.emissions_data.ram_energy,
                }
            )

        results = {
            "benchmark_name": self.name,
            "timestamp": datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            "cpu_count": general_data.cpu_count,
            "cpu_model": general_data.cpu_model,
            "gpu_count": general_data.gpu_count,
            "gpu_model": general_data.gpu_model,
            "os": general_data.os,
            "country_name": general_data.country_name,
            "country_iso_code": general_data.country_iso_code,
            "region": general_data.region,
            "python_version": general_data.python_version,
            "codecarbon_version": general_data.codecarbon_version,
            "longtitude": general_data.longitude,
            "latitude": general_data.latitude,
            "ram_total_size": general_data.ram_total_size,
            "cpu_power": general_data.cpu_power,
            "gpu_power": general_data.gpu_power,
            "ram_power": general_data.ram_power,
            "total_duration": total_duration,
            "total_cpu_energy": total_cpu_energy,
            "total_gpu_energy": total_gpu_energy,
            "total_ram_energy": total_ram_energy,
            "runs": runs,
        }

        with open("results_{}.json".format(self.name), "w", encoding='utf-8') as fp:
            json.dump(obj=results, fp=fp, indent=4)
            return results
