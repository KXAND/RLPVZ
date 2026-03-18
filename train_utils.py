import os
import time
import subprocess
from datetime import datetime

import torch

from hook_client import inject_dll
from hook_client.injector import find_pvz_process
from train_config import DEFAULT_GAME_PATH, MODEL_PATH
from callbacks import (
    MemoryResetCallback,
    AutoCollectCallback,
    SimpleMonitorCallback,
    AsyncSingleModelCallback,
    HeatmapCallback,
    DetailedLogCallback,
    DynamicEntropyCallback,
)


def launch_game_and_inject(
    game_path: str = None, wait_time: float = 3.0, port: int = 12345
) -> bool:
    """
    自动启动游戏并注入 Hook DLL

    Args:
        game_path: 游戏可执行文件路径
                                 wait_time: 启动后等待时间（秒）
        port: Hook 服务端口

    Returns:
        是否成功
    """
    if game_path is None:
        game_path = DEFAULT_GAME_PATH

    # 检查游戏是否已运行
    pid = find_pvz_process()
    if not pid:
        # 检查游戏文件是否存在
        if not os.path.exists(game_path):
            print(f"[错误] 游戏文件不存在: {game_path}")
            return False
        try:
            # 启动游戏（不等待，在后台运行）
            subprocess.Popen(
                [game_path],
                cwd=os.path.dirname(game_path),
                creationflags=subprocess.DETACHED_PROCESS,
            )
        except Exception as e:
            print(f"[错误] 启动游戏失败: {e}")
            return False

        time.sleep(wait_time)

        pid = find_pvz_process()
        if not pid:
            print("[错误] 游戏启动失败")
            return False

    # 注入 DLL
    if inject_dll():
        print(f"[Hook] DLL注入成功 (port {port})")
        # 等待 Hook 初始化
        time.sleep(1.0)
        return True
    else:
        print("[错误] DLL注入失败")
        return False


def find_latest_model():
    """查找最新的模型文件"""
    # 优先查找统一路径
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    # 兼容旧版本：搜索其他可能的模型位置
    import glob

    patterns = [
        "models/advanced_*/final_model.zip",
        "models/*/final_model.zip",
        "models/*.zip",
    ]

    all_models = []
    for pattern in patterns:
        all_models.extend(glob.glob(pattern))

    if not all_models:
        return None

    # 按修改时间排序，返回最新的
    latest = max(all_models, key=os.path.getmtime)
    return latest


def setup_logging():
    from utils.logger import get_logger, LogLevel

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logger = get_logger(level=LogLevel.DEBUG, file_path=log_file)
    print(f"\r\n[日志] 调试信息将保存到: {log_file}")
    return logger


def setup_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[设备] {torch.cuda.get_device_name(0)}")
        # 性能加速设置
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        device = "cpu"
        print("[设备] CPU")
    return device


def print_metadata(args):
    print("\r\n" + "=" * 60)
    print("高级 PVZ 训练 - 三大优化")
    print("=" * 60)

    actual_game_speed = min(args.speed, 10.0)  # tick_ms 最小为1，最高是10x
    _ = actual_game_speed * args.frameskip

    print(f"\r\n配置:")
    print(f"  速度: {actual_game_speed}x | 帧跳过: {args.frameskip} | 网络: {args.net}")
    print(f"  Batch: {args.batch} | Steps: {args.n_steps} | LR: {args.lr}")
    print(f"  探索: {args.start_ent} → {args.end_ent}")


def ensure_game_ready(args):
    if not args.no_auto_start:
        if not launch_game_and_inject(
            game_path=args.game_path, wait_time=args.wait_time, port=args.port
        ):
            print("无法启动游戏或注入 DLL，训练终止")
            print("  请使用 --no_auto_start 选项手动启动游戏和注入")
            return False
    else:
        print("自动启动已禁用，请确保游戏已启动并注入 DLL")
    return True


def resolve_load_path(args):
    """获取模型 checkpoint 路径"""
    if args.load:
        print(f"使用参数指定模型路径：{args.load}")
        return args.load

    if args.no_auto_resume:
        print("自动恢复已禁用，从零开始训练")
        return None

    load_path = find_latest_model()
    if load_path:
        print(f"自动恢复: 找到最新模型 {load_path}")
    else:
        print(f"未找到已有模型，从零开始训练")
    return load_path


def print_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU 显存: {allocated:.2f} GB")


def build_callbacks(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/advanced_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    callbacks = [
        MemoryResetCallback(verbose=0),  # 🆕 记忆重置 (在episode开始时)
        AutoCollectCallback(),  # 强制自动收集
        SimpleMonitorCallback(),  # 简洁监控：连胜/连败
        AsyncSingleModelCallback(
            save_freq=args.save_freq, save_path=args.save_path, verbose=1
        ),  # 异步保存
        HeatmapCallback(
            save_path="heatmap.html", refresh_rate=10, verbose=1
        ),  # 实时热力图 (开启 verbose 以显示 Attention Debug)
        DetailedLogCallback(log_freq=500),  # 新增：详细数据日志
    ]

    dynamic_entropy = DynamicEntropyCallback(
        start_ent_coef=args.start_ent,
        end_ent_coef=args.end_ent,
        decay_type=args.ent_decay,
        total_timesteps=args.timesteps,
        warmup_steps=min(10000, args.timesteps // 10),
        verbose=0,  # 静默
    )
    callbacks.append(dynamic_entropy)
    return callbacks


def train_model(model, env, args, callbacks):
    print(f"开始训练...\r\n")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=False,  # 关闭进度条，用自己的输出
        )
    except KeyboardInterrupt:
        print("\r\n 训练被中断")
    finally:
        model.save(args.save_path)
        print(f"\r\n模型已保存: {args.save_path}")
        env.close()
