import os
import hydra
from logging import getLogger

from omegaconf import OmegaConf, DictConfig


LOGGER = getLogger("hydra-cli")

#benchmark
@hydra.main(version_base=None, config_path="../config/", config_name="config")
def cli(config: DictConfig) -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    print(OmegaConf.to_yaml(config))

if __name__ == "__main__":
    cli()