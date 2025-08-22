"""Entry point for the multi-camera 3D human pose pipeline.

This script bootstraps configuration via Hydra, prepares output
folders, and dispatches to the Dataset or Video pipeline based on
config.mode.
"""

import hydra
from omegaconf import DictConfig

from src.pipelines.dataset import DatasetPipeline
from src.pipelines.video import VideoPipeline
from src.utils.config_utils import (
    setup_logging,
    get_all_output_paths,
)
from src.utils.file_utils import ensure_directories


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Run the selected pipeline.

    Parameters
    ----------
    config : DictConfig Hydra configuration

    Returns
    -------
    None
    """
    # Setup comprehensive logging
    logger = setup_logging(config)

    # Create output directories
    output_paths = get_all_output_paths(config)
    if output_paths:
        ensure_directories(output_paths)
        logger.info(f"Created {len(output_paths)} output directories")

    # Determine processing mode and dispatch to appropriate pipeline
    mode = getattr(config, "mode", "dataset").lower()
    logger.info(f"Running pipeline in {mode} mode")

    if mode == "video":
        pipeline = VideoPipeline(config, logger)
    else:
        pipeline = DatasetPipeline(config, logger)

    # Run the pipeline with error handling
    pipeline.run()
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
