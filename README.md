GM1: 护栏检测与运动监控系统
本项目使用 YOLOv8 实现护栏的检测，并结合 DeepSORT 进行实时物体跟踪。项目包括模型训练、推理、结果可视化和环境部署等完整流程。

目录结构:
********************************
GM1/
├── downloads/                       # 额外的下载文件目录
├── guardrail_monitoring/            # 核心项目目录
│   ├── datasets/                    # 数据集文件夹
│   │   ├── guardrail/               # 数据集：包含 images 和 labels
│   │   │   ├── images/              # 图像目录
│   │   │   ├── labels/              # 标注文件目录
│   │   │   │   ├── train/           # 训练集标注
│   │   │   │   ├── val/             # 验证集标注
│   │   │   ├── dataset.yaml         # 数据集配置文件，用于 YOLOv8
│   ├── models/                      # 自定义模型相关目录
│   ├── outputs/                     # 推理或可视化输出结果
│   ├── scripts/                     # 核心脚本目录
│   ├── utils/                       # 工具文件
│       ├── tracking.py              # DeepSORT 跟踪逻辑
│       ├── visualization.py         # 自定义可视化工具
├── runs/                            # 存储运行结果
├── environment.yml                  # Conda 环境配置文件
├── requirements.txt                 # Python 依赖文件
├── yolo11n.pt                       # 模型 1（轻量）
├── yolov8n-seg.pt                   # YOLOv8 分割模型
├── yolov8n.pt                       # YOLOv8 检测模型
**********************************


下载项目及准备工作
1. 克隆项目
git clone https://github.com/Ren-0227/ConstructionFenceTracker.git
cd GM1


2. 安装依赖
进入项目解压后所在的目录，并按以下方式安装依赖。

使用 Conda 构建环境
如果使用 Conda 管理环境，运行以下命令创建虚拟环境：
conda env create -f environment.yml
conda activate gmmm

使用 pip 安装依赖
使用 pip 管理环境的用户可以运行以下命令：
pip install -r requirements.txt

开始推理
项目内包含一系列脚本，可直接运行，开始护栏检测和跟踪任务。具体见以下说明：

1. 摄像头实时推理
运行以下命令捕获摄像头视频流并实时检测与跟踪护栏：
python scripts/infer_camera.py --model best.pt --threshold 0.25

2. 视频文件推理
将<path_to_video> 替换为你的本地视频文件路径：

python scripts/infer_video.py --model best.pt --source <path_to_video> --output outputs/

3. 图像文件推理
运行以下命令检测单张或多张图片中的护栏：
python scripts/infer_image.py --model best.pt --source <path_to_image_or_folder> --output outputs/
