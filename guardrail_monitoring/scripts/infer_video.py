import os
import argparse
from ultralytics import YOLO

def validate_paths(model_path, video_path, output_dir):
    """
    验证路径是否存在。如果路径不存在，则抛出一个错误。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径无效: {model_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频路径无效: {video_path}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"输出目录不存在，已创建: {output_dir}")

def predict_video(model_path, video_path, output_dir, conf_threshold=0.25, iou_threshold=0.5):
    """
    对视频进行 YOLO 推理并保存结果。
    """
    # 验证路径
    validate_paths(model_path, video_path, output_dir)

    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 设置输出视频保存地址
    project_dir = os.path.join(output_dir, "video_predictions")

    # 开始推理
    print(f"开始推理视频: {video_path}")
    results = model.predict(
        source=video_path,
        save=True,
        conf=conf_threshold,
        iou=iou_threshold,
        project=output_dir,
        name="video_predictions"
    )

    # 返回推理结果和保存路径
    print(f"推理完成，结果保存在: {project_dir}")
    return results

def display_results(results):
    """
    打印 YOLO 推理结果的概要信息。
    """
    print("\n推理结果：")
    for idx, result in enumerate(results):
        print(f"帧 {idx + 1}:")
        print(f"  检测框数量: {len(result.boxes)}")
        if hasattr(result, "masks") and result.masks:
            print(f"  分割掩码区域数量: {len(result.masks)}")
        else:
            print("  无分割掩码信息。")

def main():
    # 创建命令行参数
    parser = argparse.ArgumentParser(description="YOLO 视频推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--video_path", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出保存目录路径")
    parser.add_argument("--conf_threshold", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU 阈值 (默认: 0.5)")

    args = parser.parse_args()

    # 执行视频推理
    results = predict_video(
        model_path=args.model_path,
        video_path=args.video_path,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )

    # 打印结果
    display_results(results)

if __name__ == "__main__":
    main()
