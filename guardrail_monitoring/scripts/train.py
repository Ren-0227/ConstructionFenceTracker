from ultralytics import YOLO

def train_segmentation_model():
    """
    训练 YOLOv8 图像分割模型。
    """
    # 加载预训练的图像分割模型
    model = YOLO("yolov8n-seg.pt")  # 使用 YOLOv8 的图像分割模型

    # 训练模型
    results = model.train(
        data="C:/Users/ren/Desktop/gm1/guardrail_monitoring/datasets/dataset.yaml",  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        imgsz=640,   # 图像大小
        batch=16,    # 批量大小
        name="guardrail_segmentation",  # 训练任务名称
        patience=15,  # 早停机制，如果 10 轮验证集性能没有提升，则停止训练
        device="0",  # 使用 GPU 训练（如果有 GPU）
        workers=4,   # 数据加载的线程数
        optimizer="auto",  # 自动选择优化器
        lr0=0.01,    # 初始学习率
        lrf=0.01,    # 最终学习率
        weight_decay=0.0005,  # 权重衰减
        save=True,   # 保存训练结果
        save_period=10,  # 每 10 轮保存一次模型
    )

    print("训练完成！模型权重保存在 runs/segment/guardrail_segmentation/weights/ 目录下。")

if __name__ == "__main__":
    train_segmentation_model()
