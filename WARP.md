# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

This is a computer vision pipeline for multi-camera 3D human pose estimation. The system processes 2D keypoint annotations from COCO datasets through image rectification, 3D triangulation, reprojection, and evaluation phases using multiple pose estimation models (YOLO and ViTPose).

## Development Setup

### Environment Activation
```bash
source .venv/bin/activate
```

### Key Dependencies
The project uses a Python virtual environment with these main packages:
- `torch` & `torchvision` - Deep learning framework
- `ultralytics` - YOLO models for pose estimation
- `opencv-python` - Computer vision operations
- `pycocotools` - COCO dataset handling
- `scipy` - Scientific computing (for triangulation optimization)
- `tqdm` - Progress bars
- `matplotlib` - Visualization

## Common Development Commands

### Running the Main Pipeline
```bash
# Run the complete pipeline with default configuration
python main.py

# Run with specific configuration overrides
python main.py logging.level=DEBUG models.yolo.confidence_threshold=0.3

# Run only specific stages
python main.py pipeline.stages.pose_estimation.models=["yolo"]

# Use custom config files
python main.py --config-path=configs/experiments --config-name=test_config
```

### Configuration Management
```bash
# View current configuration
python main.py --cfg job

# Show configuration schema
python main.py --help

# Generate requirements.txt from current environment
pip freeze > requirements.txt

# Install dependencies
pip install -r requirements.txt hydra-core omegaconf
```

### Individual Component Testing
```bash
# Test specific modules directly
python -c "from src.utils.camera.camera_manager import CameraManager; cm = CameraManager(); print(cm)"

# Test file utilities
python -c "from src.utils.file_utils import extract_frame_number; print(extract_frame_number('out1_frame_123.jpg'))"

# Test keypoint utilities
python -c "from src.utils.keypoints import get_yolo_keypoint_names; print(len(get_yolo_keypoint_names()))"

# Test geometry utilities
python -c "from src.utils.geometry import keypoints_to_flat_array; print('Geometry utils: OK')"
```

## Architecture Overview

### Core Pipeline Flow
The main pipeline (`main.py`) orchestrates these sequential stages:

1. **Dataset Rectification** (`src/rectifier.py`)
   - Undistorts images and keypoints using camera calibration
   - Updates COCO annotations with corrected coordinates

2. **3D Triangulation** (`src/triangulator.py`) 
   - Triangulates 3D poses from 2D multi-camera observations
   - Uses Direct Linear Transform (DLT) with optional bundle adjustment
   - Outputs 3D skeleton data

3. **Pose Estimation** (Two models)
   - **YOLO**: `src/pose_estimation/yolo/pose_estimator_yolo.py`
   - **ViTPose**: `src/pose_estimation/ViTPose/pose_estimator_vitpose.py`

4. **Reprojection** (`src/reprojector.py`)
   - Projects 3D skeletons back to 2D camera views
   - Used for validation and evaluation

5. **Evaluation** (`src/utils/evaluation/pose_evaluator.py`)
   - Compares predictions against ground truth

### Key Architectural Components

#### Camera System
- **CameraManager** (`src/utils/camera/camera_manager.py`): Manages multiple camera calibrations
- **CameraCalibration** (`src/utils/camera/camera_calibration.py`): Individual camera parameters
- Expected structure: `data/camera_data_v2/cam_*/calib/camera_calib.json`

#### Dataset Management
- **COCOManager** (`src/utils/dataset/coco_utils.py`): COCO dataset manipulation with keypoint pruning
- **SkeletonManager3D** (`src/utils/dataset/skeleton_3d.py`): 3D skeleton data management

#### Visualization
- **SkeletonDrawer** (`src/utils/skeleton/pose_plotter_2d.py`): 2D pose visualization
- **PosePlotter3D** (`src/utils/skeleton/pose_plotter_3d.py`): 3D pose animation

### Data Flow Patterns

1. **Multi-Camera Input**: Images follow naming pattern `out{cam_id}_frame_{frame_num}.jpg`
2. **COCO Format**: All 2D keypoints stored in COCO JSON format
3. **Structured Output**: Organized by method (yolo/vit) and type (predictions/triangulations/reprojections/evaluations)

### Important Directory Structure

```
├── data/
│   ├── camera_data_v2/        # Camera calibrations (cam_0/, cam_1/, etc.)
│   └── dataset/               # Input images + _annotations.coco.json
├── src/
│   ├── pose_estimation/       # YOLO and ViTPose estimators
│   ├── utils/
│   │   ├── camera/           # Camera management
│   │   ├── dataset/          # COCO and skeleton utilities
│   │   ├── evaluation/       # Pose evaluation
│   │   └── skeleton/         # Visualization tools
│   ├── triangulator.py       # 3D reconstruction
│   ├── rectifier.py          # Image/annotation rectification  
│   └── reprojector.py        # 3D to 2D reprojection
├── weights/                   # YOLO model weights
└── output/                    # Generated results (auto-created)
```

## Configuration System (Hydra)

The codebase uses Hydra for configuration management, providing flexible and composable configuration.

### Configuration Structure
```
configs/
├── config.yaml          # Main config with defaults
├── paths/
│   └── default.yaml     # File paths and directories
├── models/
│   └── default.yaml     # Model settings (YOLO, ViT, triangulation)
├── pipeline/
│   └── default.yaml     # Pipeline execution settings
└── logging/
    └── default.yaml     # Logging configuration
```

### Key Configuration Patterns
- **Modular configs**: Organized by component (paths, models, pipeline, logging)
- **Override capability**: Command-line parameter overrides
- **Environment support**: Different configs for different environments
- **Type safety**: OmegaConf provides structured configuration with type hints

### Common Configuration Overrides
```bash
# Change model selection
python main.py models.pose_estimators=["yolo"]

# Adjust confidence thresholds
python main.py models.yolo.confidence_threshold=0.5

# Enable/disable pipeline stages
python main.py pipeline.stages.rectification.enabled=false

# Change output paths
python main.py paths.output.base_dir="/custom/output"

# Logging configuration
python main.py logging.level=DEBUG logging.file.enabled=false
```

## Development Patterns

### Code Organization
- **File utilities**: File/path operations in `src/utils/file_utils.py`
- **Validation utilities**: Path validation and input checking in `src/utils/validation.py`
- **Configuration utilities**: Hydra config management and logging in `src/utils/config_utils.py`
- **Geometry utilities**: Coordinate transformations and bounding boxes in `src/utils/geometry.py`
- **Keypoint utilities**: Pose model keypoint mapping and virtual keypoints in `src/utils/keypoints.py`
- **Semantic organization**: Functions grouped by purpose rather than usage frequency
- **No code duplication**: Centralized utilities with clear responsibilities

### Error Handling
- Input validation through Hydra configuration validation
- Path existence checks before pipeline execution
- Graceful handling of missing images with logging
- Type hints and structured configuration prevent runtime errors

### Keypoint Handling
- Configurable keypoint pruning via `config.keypoints.prune_patterns`
- Centralized virtual keypoint calculation (Hips, Neck, Spine)
- COCO format: `[x1,y1,v1,x2,y2,v2,...]` where `v` is visibility
- Keypoint mapping abstracted for different models (YOLO vs ViTPose)

### Bundle Adjustment
Configurable via `config.models.triangulation.use_bundle_adjustment` with optimization method selection.

### Performance Considerations
- Pre-computed undistortion maps for rectification efficiency
- Progress bars via `tqdm` for long-running operations  
- Memory-efficient COCO dataset processing
- Configurable batch processing parameters

## Model Integration Notes

### YOLO Integration
- Uses `ultralytics` package
- Supports confidence thresholding
- Maps YOLO's 17-keypoint format to custom keypoint schema

### ViTPose Integration  
- Requires HuggingFace model (`usyd-community/vitpose-plus-base`)
- Uses YOLO for person detection, ViTPose for keypoint estimation
- Handles model loading and inference pipeline

## File Naming Conventions
- Camera files: `out{cam_id}_frame_{frame_num}.jpg`
- Output files: Organized by algorithm and data type
- JSON annotations: Standard COCO format with extensions for 3D data
