"""Main pipeline for multi-camera 3D human pose estimation."""

import hydra
from omegaconf import DictConfig

from src.pipelines.dataset import DatasetPipeline
from src.pipelines.video import VideoPipeline
from src.utils.config_utils import setup_logging, get_all_output_paths
from src.utils.file_utils import ensure_directories



@hydra.main(version_base=None, config_path="configs", config_name="config_dataset")
def main(config: DictConfig) -> None:
    """Main pipeline execution function.
    
    Args:
        config: Hydra configuration object
    """
    # Setup logging
    logger = setup_logging()
    
    # Create output directories
    output_paths = get_all_output_paths(config)
    ensure_directories(output_paths)
    
    # Determine processing mode and dispatch to appropriate pipeline
    mode = getattr(config, "mode", "dataset").lower()
    
    if mode == "video":
        pipeline = VideoPipeline(config, logger)
    else:
        pipeline = DatasetPipeline(config, logger)
    
    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    main()