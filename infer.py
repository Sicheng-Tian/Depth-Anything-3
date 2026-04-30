import glob
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

# 添加 src 目录到 Python 路径
project_root = Path(__file__).parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import export_to_glb

# ========== 配置参数 ==========
MODEL_NAME = "da3-giant-mask"  # 使用嵌套模型配置
CHECKPOINT_PATH = "checkpoints/da3-mask-8/model.safetensors"

device = torch.device("cuda")

# ========== 时间统计 ==========
timing_stats = {}
total_start = time.time()

# 初始化模型
print(f"正在初始化模型: {MODEL_NAME}")
start_time = time.time()
model = DepthAnything3(model_name=MODEL_NAME)
timing_stats["模型初始化"] = time.time() - start_time

# 加载权重
print(f"正在加载checkpoint: {CHECKPOINT_PATH}")
start_time = time.time()
state_dict = load_file(CHECKPOINT_PATH)
timing_stats["加载checkpoint文件"] = time.time() - start_time

# 处理权重键名（移除 'model.' 前缀）
start_time = time.time()
for k in list(state_dict.keys()):
    if k.startswith("model."):
        state_dict[k[6:]] = state_dict.pop(k)
timing_stats["处理权重键名"] = time.time() - start_time

# 加载权重到模型
start_time = time.time()
model.model.load_state_dict(state_dict, strict=False)
timing_stats["加载权重到模型"] = time.time() - start_time
print("✓ 模型权重加载完成")

# 移动到 GPU
start_time = time.time()
model = model.to(device=device)
model.model.eval()
timing_stats["移动模型到GPU"] = time.time() - start_time

# 加载示例图片
start_time = time.time()
example_path = "/mnt/disk4.5-2/WL/Depth-Anything-3gshead/assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
timing_stats["加载图片列表"] = time.time() - start_time
print(f"正在处理 {len(images)} 张图片...")

# ========== 运行推理 ==========
OUTPUT_DIR = "visualizations/outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 推理阶段（启用 3DGS 输出）
start_time = time.time()
prediction = model.inference(
    images,
    export_dir=None,  # 不自动导出，手动控制
    infer_gs=True,  # 启用 3DGS 分支
)
timing_stats["模型推理"] = time.time() - start_time

# 2. 导出 GLB 点云
start_time = time.time()
export_to_glb(
    prediction,
    export_dir=OUTPUT_DIR,
    conf_thresh_percentile=20.0,
    num_max_points=1_000_000,
    show_cameras=True,
)
timing_stats["导出GLB文件"] = time.time() - start_time

# 3. 导出 3DGS PLY
if prediction.gaussians is not None:
    from depth_anything_3.utils.gsply_helpers import save_gaussian_ply

    print("正在导出 3DGS...")
    start_time = time.time()

    gs_dir = os.path.join(OUTPUT_DIR, "gs_ply")
    os.makedirs(gs_dir, exist_ok=True)

    gs_world = prediction.gaussians
    pred_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(gs_world.means)

    # 保存主 PLY（所有视图的 Gaussians 合并）
    ply_path = os.path.join(gs_dir, "0000.ply")
    save_gaussian_ply(
        gaussians=gs_world,
        save_path=ply_path,
        ctx_depth=pred_depth,
        shift_and_scale=False,
        save_sh_dc_only=True,
        gs_views_interval=max(pred_depth.shape[0] // 12, 1),
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )
    timing_stats["导出3DGS_PLY"] = time.time() - start_time
    print(f"✓ 3DGS PLY 已保存: {ply_path}")
else:
    print("⚠ 警告: prediction.gaussians 为空，3DGS 不可用")
    print("  可能原因: 模型配置中未包含 gs_head / gs_adapter")

# ========== 输出结果信息 ==========
print("\n预测结果:")
print(f"  processed_images 形状: {prediction.processed_images.shape}")
print(f"  depth 形状: {prediction.depth.shape}")
print(f"  conf 形状: {prediction.conf.shape}")
print(f"  extrinsics 形状: {prediction.extrinsics.shape}")
print(f"  intrinsics 形状: {prediction.intrinsics.shape}")
if prediction.gaussians is not None:
    n_gaussians = prediction.gaussians.means.shape[0]
    print(f"  3DGS Gaussians 数量: {n_gaussians:,}")

# ========== 导出完成 ==========
print(f"\n✓ 输出文件已导出到: {OUTPUT_DIR}")
print("  - scene.glb         (3D 点云场景，含相机)")
print("  - scene.jpg         (深度可视化图像)")
print("  - depth_vis/        (所有帧的深度可视化)")
print("  - gs_ply/0000.ply   (3DGS Gaussian PLY 文件)")

print("\n📊 GLB 文件在线查看器:")
print("  1. https://gltf-viewer.donmccarll.com/")
print("  2. https://sandbox.babylonjs.com/")
print("  3. https://threejs.org/editor/")

print("\n🔄 3DGS PLY 查看工具:")
print("  推荐使用 SIBR 或 SAFS (3D Gaussian Splatting 可视化工具)")

# ========== 打印时间统计 ==========
timing_stats["总耗时"] = time.time() - total_start

print("\n" + "=" * 60)
print("⏱  各步骤耗时统计")
print("=" * 60)
for step_name, elapsed_time in timing_stats.items():
    if step_name == "总耗时":
        continue
    print(f"  {step_name:<20s} : {elapsed_time:>8.3f} 秒")
print("-" * 60)
print(f"  {'总耗时':<20s} : {timing_stats['总耗时']:>8.3f} 秒")
print("=" * 60)
