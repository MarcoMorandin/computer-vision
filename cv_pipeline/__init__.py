"""Computer Vision Pipeline

A modular computer vision pipeline for multi-view human pose estimation and 3D reconstruction.
"""

__version__ = "1.0.0"
__author__ = "Marco Morandin"

from .core.pipeline import CVPipeline
from .data_sources import RoboflowDataSource, VideoFileDataSource, StreamingVideoDataSource
from .rectification import CameraRectifier
from .pose_estimation import YOLOPoseEstimator, ViTPoseEstimator, ProbPoseEstimator, MediaPipePoseEstimator
from .triangulation import MultiViewTriangulator
from .reprojection import ReprojectionEvaluator

__all__ = [
    'CVPipeline',
    'RoboflowDataSource',
    'VideoFileDataSource', 
    'StreamingVideoDataSource',
    'CameraRectifier',
    'YOLOPoseEstimator',
    'ViTPoseEstimator',
    'ProbPoseEstimator',
    'MediaPipePoseEstimator',
    'MultiViewTriangulator',
    'ReprojectionEvaluator'
]
