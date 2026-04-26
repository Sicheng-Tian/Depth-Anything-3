# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# inference script for Depth Anything 3 - Image/Video to 3DGS

"""
DA3 Inference Script: Image/Video Directory -> 3D Gaussian Splatting

指定输入路径、输出路径和模型路径，调用 DA3 模型进行深度估计和3DGS生成。

功能:
  - 输入: 单张图片 / 图片目录 / 视频文件
  - 模型: 支持本地模型路径或 HuggingFace 模型ID
  - 输出: 3DGS PLY文件 / GLB点云 / 渲染视频 / 深度图可视化

用法示例:
    # 1. 使用本地 DA3-Giant checkpoint (默认使用 da3-giant 配置):
    python inference.py -i ./images -o ./output -m /path/to/da3-giant/

    # 2. 使用本地 checkpoint，指定不同的配置文件:
    python inference.py -i ./images -o ./output -m /path/to/da3-giant/ -c da3-large

    # 3. 使用 HuggingFace 模型 (推荐):
    python inference.py -i ./images -o ./output -m depth-anything/DA3NESTED-GIANT-LARGE-1.1

    # 4. 处理单张图片:
    python inference.py -i ./test.jpg -o ./output -m depth-anything/DA3-LARGE-1.1

    # 5. 处理视频文件:
    python inference.py -i ./video.mp4 -o ./output -m depth-anything/DA3-LARGE-1.1 --video_fps 2.0

    # 6. 调整处理分辨率:
    python inference.py -i ./images -o ./output -m depth-anything/DA3-LARGE-1.1 --process_res 756

    # 7. 仅导出 PLY 点云 (不渲染视频, 节省显存):
    python inference.py -i ./images -o ./output -m depth-anything/DA3-LARGE-1.1 -f gs_ply

    # 8. 导出多个格式:
    python inference.py -i ./images -o ./output -m depth-anything/DA3-LARGE-1.1 -f "gs_ply,glb,depth_vis"

    # 可用 --config 选项: da3-giant(默认), da3-large, da3-base, da3-small,
    #                      da3metric-large, da3mono-large, da3nested-giant-large
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import torch

# Add src to path for development use
_SRC_DIR = Path(__file__).parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.logger import logger

# ============================================================
# Supported formats
# ============================================================
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}

# Default model config (used when local checkpoint has no config.json)
DEFAULT_CONFIG_NAME = "da3-giant"

# Available YAML configs bundled in the package
BUNDLED_CONFIGS = {
    "da3-giant": "depth_anything_3.configs.da3-giant",
    "da3-large": "depth_anything_3.configs.da3-large",
    "da3-base": "depth_anything_3.configs.da3-base",
    "da3-small": "depth_anything_3.configs.da3-small",
    "da3metric-large": "depth_anything_3.configs.da3metric-large",
    "da3mono-large": "depth_anything_3.configs.da3mono-large",
    "da3nested-giant-large": "depth_anything_3.configs.da3nested-giant-large",
}

# Short aliases for HuggingFace model IDs
AVAILABLE_MODELS = {
    "da3-nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "da3-giant": "depth-anything/DA3-GIANT-1.1",
    "da3-large": "depth-anything/DA3-LARGE-1.1",
    "da3-base": "depth-anything/DA3-BASE",
    "da3-small": "depth-anything/DA3-SMALL",
    "da3metric-large": "depth-anything/DA3METRIC-LARGE",
}


# ============================================================
# Model loading
# ============================================================
def load_model(
    model_path: str,
    config_name: str = DEFAULT_CONFIG_NAME,
    device: str = "cuda",
) -> DepthAnything3:
    """Load DA3 model from local path or HuggingFace Hub.

    For local checkpoints without config.json, uses the bundled YAML config
    to build the model architecture, then loads the safetensors weights.

    Args:
        model_path: Local directory path or HuggingFace model ID
        config_name: YAML config name to use for local checkpoints.
                    Options: da3-giant (default), da3-large, da3-base, da3-small,
                    da3metric-large, da3mono-large, da3nested-giant-large.
                    Can also be a path to a custom .yaml file.
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        Loaded DepthAnything3 model
    """
    logger.info(f"Loading model from: {model_path}")

    load_start = time.time()
    if os.path.exists(model_path):
        config_json = os.path.join(model_path, "config.json")
        safetensors_file = os.path.join(model_path, "model.safetensors")

        if os.path.exists(config_json):
            logger.info("Found config.json, using from_pretrained")
            model = DepthAnything3.from_pretrained(model_path)
        else:
            logger.info(f"No config.json found, using config: {config_name}")
            model = _build_model_from_yaml_and_load_weights(
                model_path, config_name, safetensors_file
            )
    else:
        hf_path = AVAILABLE_MODELS.get(model_path, model_path)
        logger.info(f"Loading from HuggingFace: {hf_path}")
        model = DepthAnything3.from_pretrained(hf_path)

    model = model.to(device)
    model.eval()

    load_time = time.time() - load_start
    logger.info(f"Model loaded in {load_time:.1f}s on {device}")
    return model


def _build_model_from_yaml_and_load_weights(
    model_dir: str,
    config_name: str,
    safetensors_path: str,
) -> DepthAnything3:
    """Build model from YAML config, then load weights from safetensors."""
    import safetensors.torch

    from depth_anything_3.cfg import load_config
    from depth_anything_3.model.da3 import DepthAnything3Net, NestedDepthAnything3Net

    config_path = config_name
    if config_name in BUNDLED_CONFIGS:
        config_path = BUNDLED_CONFIGS[config_name]

    logger.info(f"Loading config: {config_path}")
    cfg = load_config(config_path)

    obj_cfg = cfg.get("__object__", {})
    model_cls_name = obj_cfg.get("name", "")

    if model_cls_name == "NestedDepthAnything3Net":
        model = NestedDepthAnything3Net(cfg.get("anyview"), cfg.get("metric"))
    else:
        model = DepthAnything3Net(
            net=cfg.get("net"),
            head=cfg.get("head"),
            cam_dec=cfg.get("cam_dec"),
            cam_enc=cfg.get("cam_enc"),
            gs_head=cfg.get("gs_head"),
            gs_adapter=cfg.get("gs_adapter"),
        )

    if os.path.exists(safetensors_path):
        logger.info(f"Loading weights from: {safetensors_path}")
        state_dict = safetensors.torch.load_file(safetensors_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warn(f"Missing keys: {missing}")
        if unexpected:
            logger.warn(f"Unexpected keys (ignored): {unexpected}")
    else:
        raise FileNotFoundError(
            f"Safetensors file not found: {safetensors_path}\n"
            f"Available files in {model_dir}: {os.listdir(model_dir)}"
        )

    wrapped = DepthAnything3(model_name=config_name)
    wrapped.model = model
    return wrapped


# ============================================================
# Input path detection
# ============================================================
def detect_input_type(input_path: str) -> str:
    """Detect whether input is a single image, image directory, or video."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in IMAGE_EXTS:
            return "image"
        elif ext in VIDEO_EXTS:
            return "video"
        else:
            raise ValueError(
                f"Unsupported file type: {ext}. Supported: {IMAGE_EXTS | VIDEO_EXTS}"
            )

    if os.path.isdir(input_path):
        for item in os.listdir(input_path):
            item_path = os.path.join(input_path, item)
            if os.path.isfile(item_path):
                ext = os.path.splitext(item)[1].lower()
                if ext in IMAGE_EXTS:
                    return "images"
        raise ValueError(f"No image files found in directory: {input_path}")

    return "unknown"


def collect_image_files(input_path: str) -> list[str]:
    """Collect all image file paths from a directory, sorted by filename."""
    image_files = []
    for item in sorted(os.listdir(input_path)):
        item_path = os.path.join(input_path, item)
        if os.path.isfile(item_path):
            ext = os.path.splitext(item)[1].lower()
            if ext in IMAGE_EXTS:
                image_files.append(item_path)
    return image_files


def extract_frames_from_video(
    video_path: str, output_dir: str, fps: float = 1.0
) -> list[str]:
    """Extract frames from video at specified FPS.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract

    Returns:
        List of extracted frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Video: {video_fps:.1f}fps, {total_frames} frames, extracting at {fps}fps"
    )

    frame_interval = max(1, int(round(video_fps / fps)))
    frame_idx = 0
    saved_paths = []

    pbar_desc = "Extracting frames"
    try:
        from tqdm import tqdm

        pbar = tqdm(total=int(total_frames / frame_interval), desc=pbar_desc)
    except ImportError:
        pbar = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            save_path = os.path.join(output_dir, f"frame_{len(saved_paths):06d}.png")
            cv2.imwrite(save_path, frame)
            saved_paths.append(save_path)
            if pbar:
                pbar.update(1)

        frame_idx += 1

    cap.release()
    if pbar:
        pbar.close()

    logger.info(f"Extracted {len(saved_paths)} frames to {output_dir}")
    return saved_paths


# ============================================================
# Output directory preparation
# ============================================================
def prepare_output_dir(output_path: str, clean: bool = False) -> str:
    """Create and prepare output directory."""
    output_path = os.path.abspath(output_path)

    if os.path.exists(output_path):
        if clean:
            logger.info(f"Cleaning existing output directory: {output_path}")
            shutil.rmtree(output_path)
        elif os.path.isfile(output_path):
            raise ValueError(f"Output path exists as a file: {output_path}")
        elif os.listdir(output_path):
            logger.info(
                f"Output directory not empty, appending timestamp: {output_path}"
            )
            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, f"da3_output_{ts}")
            logger.info(f"New output directory: {output_path}")

    os.makedirs(output_path, exist_ok=True)
    return output_path


# ============================================================
# Export helpers
# ============================================================
def export_gs_ply_from_prediction(prediction, export_dir: str) -> str:
    """Export 3DGS PLY file from Prediction object.

    This saves the 3D Gaussian Splatting parameters as a PLY file that can be
    viewed in tools like 3D Gaussian Splatting viewers (e.g., SIBR, SAFS).
    """
    from depth_anything_3.utils.export.gs import export_to_gs_ply
    from depth_anything_3.utils.gsply_helpers import save_gaussian_ply

    export_to_gs_ply(prediction, export_dir)
    ply_path = os.path.join(export_dir, "gs_ply", "0000.ply")
    if os.path.exists(ply_path):
        return ply_path

    gs_world = prediction.gaussians
    pred_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(gs_world.means)
    gs_dir = os.path.join(export_dir, "gs_ply")
    os.makedirs(gs_dir, exist_ok=True)
    save_path = os.path.join(gs_dir, "0000.ply")
    save_gaussian_ply(
        gaussians=gs_world,
        save_path=save_path,
        ctx_depth=pred_depth,
        shift_and_scale=False,
        save_sh_dc_only=True,
        gs_views_interval=max(pred_depth.shape[0] // 12, 1),
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )
    return save_path


def export_glb(
    prediction, export_dir: str, conf_thresh_percentile: float = 40.0
) -> str:
    """Export GLB point cloud from Prediction object."""
    from depth_anything_3.utils.export.glb import export_to_glb

    export_to_glb(
        prediction,
        export_dir,
        conf_thresh_percentile=conf_thresh_percentile,
        num_max_points=1_000_000,
        show_cameras=True,
    )
    glb_path = os.path.join(export_dir, "scene.glb")
    return glb_path


def export_depth_vis(prediction, export_dir: str) -> list[str]:
    """Export depth map visualizations."""
    from depth_anything_3.utils.export.depth_vis import export_to_depth_vis

    export_to_depth_vis(prediction, export_dir)
    depth_dir = os.path.join(export_dir, "depth_vis")
    paths = []
    if os.path.exists(depth_dir):
        for f in sorted(os.listdir(depth_dir)):
            if f.endswith((".png", ".jpg")):
                paths.append(os.path.join(depth_dir, f))
    return paths


def export_npz(prediction, export_dir: str) -> str:
    """Export NPZ with depth, poses, intrinsics."""
    from depth_anything_3.utils.export.npz import export_to_mini_npz

    export_to_mini_npz(prediction, export_dir)
    npz_path = os.path.join(export_dir, "prediction.mini.npz")
    return npz_path


def export_gs_video(
    prediction,
    export_dir: str,
    trj_mode: str = "smooth",
    video_quality: str = "medium",
) -> list[str]:
    """Export 3DGS rendered video.

    Requires gsplat package:
        pip install git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

    Trajectory modes:
        - "wander": Smooth wandering camera path (good for single/dual views)
        - "smooth": Smooth interpolation along input camera path
        - "interpolate": Linear interpolation between views
        - "interpolate_smooth": Smoothed interpolation
        - "extend": Extended path following input trajectory
        - "dolly_zoom": Dolly zoom effect
        - "wobble_inter": Wobble interpolation
    """
    try:
        from depth_anything_3.utils.export.gs import export_to_gs_video
    except ImportError as e:
        logger.warn(f"gs export not available: {e}")
        return []

    export_to_gs_video(
        prediction,
        export_dir,
        trj_mode=trj_mode,
        video_quality=video_quality,
    )
    gs_video_dir = os.path.join(export_dir, "gs_video")
    paths = []
    if os.path.exists(gs_video_dir):
        for f in sorted(os.listdir(gs_video_dir)):
            if f.endswith(".mp4"):
                paths.append(os.path.join(gs_video_dir, f))
    return paths


# ============================================================
# Main inference pipeline
# ============================================================
def run_inference(
    input_path: str,
    output_path: str,
    model_path: str,
    config_name: str = DEFAULT_CONFIG_NAME,
    device: str = "cuda",
    export_format: str = "gs_ply,glb,depth_vis",
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    video_fps: float = 1.0,
    conf_thresh_percentile: float = 40.0,
    num_max_points: int = 1_000_000,
    show_cameras: bool = True,
    gs_video_quality: str = "medium",
    gs_trj_mode: str = "smooth",
    clean_output: bool = False,
) -> dict:
    """Run full DA3 inference pipeline.

    Args:
        input_path: Path to input (image file, image directory, or video file)
        output_path: Path to output directory
        model_path: Local model path or HuggingFace model ID
        config_name: YAML config for building local model architecture.
                   Options: da3-giant (default), da3-large, da3-base, da3-small,
                   da3metric-large, da3mono-large, da3nested-giant-large.
                   Or path to a custom .yaml file.
        device: Device to run inference on ("cuda" or "cpu")
        export_format: Comma-separated export formats.
                      Options: gs_ply, gs_video, glb, depth_vis, npz
                      Use "all" to export everything.
        process_res: Processing resolution (default 504)
        process_res_method: Resize method - "upper_bound_resize" or "lower_bound_resize"
        use_ray_pose: Use ray-based pose estimation (slower, more accurate)
        ref_view_strategy: Reference view selection strategy.
                          Options: "first", "middle", "saddle_balanced", "saddle_sim_range"
        video_fps: FPS for extracting frames from video input
        conf_thresh_percentile: Confidence threshold percentile for GLB export
        num_max_points: Max point cloud size for GLB export
        show_cameras: Show camera wireframes in GLB export
        gs_video_quality: Quality for gs_video export ("low", "medium", "high")
        gs_trj_mode: Trajectory mode for gs_video ("wander", "smooth", "interpolate", etc.)
        clean_output: Clean output directory before writing

    Returns:
        Dictionary containing inference results and export paths
    """
    total_start = time.time()

    # 1. Prepare output directory
    output_path = prepare_output_dir(output_path, clean=clean_output)
    logger.info(f"Output directory: {output_path}")

    # 2. Detect input type and collect image paths
    input_type = detect_input_type(input_path)
    logger.info(f"Detected input type: {input_type}")

    frames_dir = None
    if input_type == "image":
        image_paths = [os.path.abspath(input_path)]
        logger.info(f"Processing single image: {image_paths[0]}")
    elif input_type == "images":
        image_paths = [os.path.abspath(p) for p in collect_image_files(input_path)]
        logger.info(f"Processing {len(image_paths)} images from: {input_path}")
    elif input_type == "video":
        frames_dir = os.path.join(output_path, "_temp_frames")
        image_paths = extract_frames_from_video(input_path, frames_dir, fps=video_fps)
        logger.info(f"Extracted {len(image_paths)} frames from video")
    else:
        raise RuntimeError(f"Unknown input type: {input_type}")

    if len(image_paths) == 0:
        raise ValueError("No images to process")

    # 3. Load model
    logger.info("=" * 60)
    model = load_model(model_path, config_name=config_name, device=device)

    # 4. Determine export formats
    export_formats = [f.strip() for f in export_format.split(",")]
    if "all" in export_formats:
        export_formats = ["gs_ply", "gs_video", "glb", "depth_vis", "npz"]

    # Check if 3DGS export is requested
    needs_gs = "gs_ply" in export_formats or "gs_video" in export_formats

    # 5. Run inference
    logger.info("=" * 60)
    logger.info(f"Running inference on {len(image_paths)} images...")
    logger.info(f"Export formats: {export_formats}")
    if needs_gs:
        logger.info("3DGS inference ENABLED")
    logger.info("=" * 60)

    # Build export kwargs for gs_video parameters
    export_kwargs = {}
    if needs_gs:
        export_kwargs["gs_video"] = {
            "trj_mode": gs_trj_mode,
            "video_quality": gs_video_quality,
        }

    inference_start = time.time()
    prediction = model.inference(
        image=image_paths,
        export_dir=output_path,
        export_format="-".join(export_formats),
        process_res=process_res,
        process_res_method=process_res_method,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
        infer_gs=needs_gs,
        conf_thresh_percentile=conf_thresh_percentile,
        num_max_points=num_max_points,
        show_cameras=show_cameras,
        export_kwargs=export_kwargs,
    )
    inference_time = time.time() - inference_start
    logger.info(f"Inference completed in {inference_time:.1f}s")

    # 6. Summary
    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Processed images: {len(image_paths)}")
    logger.info(f"Depth shape: {prediction.depth.shape}")
    if prediction.extrinsics is not None:
        logger.info(f"Camera poses: {len(prediction.extrinsics)} views")
    logger.info(f"Metric depth: {'Yes' if prediction.is_metric else 'No'}")
    if prediction.gaussians is not None:
        logger.info(f"3DGS Gaussians: {prediction.gaussians.means.shape[0]} points")
    logger.info("=" * 60)

    # 7. Print export paths
    logger.info("\nExported files:")
    for root, dirs, files in os.walk(output_path):
        for f in sorted(files):
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, output_path)
            size_kb = os.path.getsize(full_path) / 1024
            if size_kb > 1024:
                logger.info(f"  {rel_path} ({size_kb / 1024:.1f} MB)")
            else:
                logger.info(f"  {rel_path} ({size_kb:.1f} KB)")

    # 8. Cleanup temp frames
    if frames_dir and os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
        logger.info("Cleaned up temporary frame extraction directory")

    return {
        "output_path": output_path,
        "prediction": prediction,
        "image_count": len(image_paths),
        "inference_time": inference_time,
        "total_time": total_time,
    }


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="DA3 Inference: Image/Video to 3DGS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        required=True,
        help="Path to input: image file, image directory, or video file",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
        help="Model path: local directory or HuggingFace model ID "
        "(e.g., 'depth-anything/DA3-LARGE-1.1', 'da3-nested-giant-large', "
        "'da3-giant', 'da3-large', 'da3-base', 'da3-small')",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        choices=list(BUNDLED_CONFIGS.keys()),
        help="YAML config for local checkpoint architecture. "
        f"Options: {list(BUNDLED_CONFIGS.keys())}. "
        "Or a path to a custom .yaml file. (default: da3-giant)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--export_format",
        "-f",
        type=str,
        default="gs_ply,glb,depth_vis",
        help="Comma-separated export formats: gs_ply, gs_video, glb, depth_vis, npz, all "
        "(default: gs_ply,glb,depth_vis)",
    )
    parser.add_argument(
        "--process_res",
        type=int,
        default=504,
        help="Processing resolution (default: 504). "
        "Higher values = better quality but slower.",
    )
    parser.add_argument(
        "--process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
        help="Image resize method (default: upper_bound_resize)",
    )
    parser.add_argument(
        "--use_ray_pose",
        action="store_true",
        help="Use ray-based pose estimation (slower but potentially more accurate)",
    )
    parser.add_argument(
        "--ref_view_strategy",
        type=str,
        default="saddle_balanced",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
        help="Reference view selection strategy for multi-view processing "
        "(default: saddle_balanced)",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=1.0,
        help="FPS for extracting frames from video input (default: 1.0)",
    )
    parser.add_argument(
        "--conf_thresh_percentile",
        type=float,
        default=40.0,
        help="Confidence threshold percentile for GLB export (default: 40.0)",
    )
    parser.add_argument(
        "--num_max_points",
        type=int,
        default=1_000_000,
        help="Maximum number of points in point cloud export (default: 1000000)",
    )
    parser.add_argument(
        "--show_cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show camera wireframes in GLB export (default: True)",
    )
    parser.add_argument(
        "--gs_video_quality",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Quality for gs_video export (default: medium)",
    )
    parser.add_argument(
        "--gs_trj_mode",
        type=str,
        default="smooth",
        choices=[
            "wander",
            "smooth",
            "interpolate",
            "interpolate_smooth",
            "extend",
            "dolly_zoom",
            "wobble_inter",
        ],
        help="Camera trajectory mode for gs_video export (default: smooth). "
        "'wander' is recommended for single/dual view inputs.",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clean output directory before writing",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warn("CUDA not available, falling back to CPU")
        args.device = "cpu"

    run_inference(
        input_path=args.input_path,
        output_path=args.output_path,
        model_path=args.model_path,
        config_name=args.config,
        device=args.device,
        export_format=args.export_format,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        video_fps=args.video_fps,
        conf_thresh_percentile=args.conf_thresh_percentile,
        num_max_points=args.num_max_points,
        show_cameras=args.show_cameras,
        gs_video_quality=args.gs_video_quality,
        gs_trj_mode=args.gs_trj_mode,
        clean_output=args.clean,
    )


if __name__ == "__main__":
    main()
