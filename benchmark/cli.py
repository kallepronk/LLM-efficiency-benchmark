import os
import hydra
from logging import getLogger

from omegaconf import OmegaConf, DictConfig

from benchmark.main import Benchmark

LOGGER = getLogger("hydra-cli")

#benchmark
@hydra.main(version_base=None, config_path="../config/", config_name="config")
def cli(config: DictConfig) -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    print(OmegaConf.to_yaml(config))
    benchmark: Benchmark = hydra.utils.instantiate(config)
    print(benchmark.__dict__)

if __name__ == "__main__":
    cli()