import os
import argparse
from ultralytics import YOLO

def validate_paths(model_path, validation_dir, output_dir):
    """验证路径是否存在"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径无效: {model_path}")
    if not os.path.exists(validation_dir):
        raise FileNotFoundError(f"验证集路径无效: {validation_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"输出目录不存在，已创建: {output_dir}")

def predict_guardrail_images(model_path, validation_dir, output_dir, conf_threshold=0.25, iou_threshold=0.5):
    """使用 YOLO 模型对验证集进行推理，并保存结果"""
    # 验证路径
    validate_paths(model_path, validation_dir, output_dir)

    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 设置推理参数，并执行预测
    print(f"开始推理，验证集路径: {validation_dir}")
    results = model.predict(
        source=validation_dir,  # 输入验证集路径
        save=True,             # 保存结果
        conf=conf_threshold,   # 置信度阈值
        iou=iou_threshold,     # IoU 阈值
        project=output_dir,    # 项目保存路径
        name="predictions"     # 子目录名
    )

    # 返回推理结果
    return results

def display_results(results):
    """打印 YOLO 推理结果的关键信息"""
    print("\n推理完成，结果信息如下：")
    for idx, result in enumerate(results):
        print(f"图像 {idx + 1}:")
        print(f"  检测框数量: {len(result.boxes)}")
        if hasattr(result, "masks") and result.masks is not None:
            print(f"  分割掩码数量: {len(result.masks)}")
        else:
            print("  无分割掩码信息")

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="YOLO 图像推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的 YOLO 模型路径")
    parser.add_argument("--validation_dir", type=str, required=True, help="验证集图片路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出结果保存路径")
    parser.add_argument("--conf_threshold", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU 阈值 (默认: 0.5)")

    args = parser.parse_args()

    # 执行推理
    results = predict_guardrail_images(
        model_path=args.model_path,
        validation_dir=args.validation_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )

    # 显示结果
    display_results(results)

if __name__ == "__main__":
    main()
