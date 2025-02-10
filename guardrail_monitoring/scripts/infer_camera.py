import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def main(args):
    # 初始化 YOLO 模型
    model = YOLO(args.model_path)

    # 初始化 DeepSORT
    deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100, 
                        max_iou_distance=0.7, max_cosine_distance=0.2)

    # 打开摄像头
    cap = cv2.VideoCapture(args.camera_index)

    # 检查摄像头是否打开
    if not cap.isOpened():
        print(f"错误：无法打开摄像头（索引：{args.camera_index}）！请检查摄像头连接或输入正确索引。")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 视频保存设置
    if args.save_video:
        print(f"保存视频到：{args.output_file}")
        out = cv2.VideoWriter(args.output_file, 
                              cv2.VideoWriter_fourcc(*"mp4v"), 
                              fps, 
                              (frame_width, frame_height))

    print("开始实时检测和跟踪，按 'q' 键退出...")

    # 主检测/跟踪循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头视频读取结束")
            break

        # YOLO 检测
        results = model(frame)

        # 存储检测框 (xmin, ymin, xmax, ymax)、置信度和类别
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                if conf > args.conf_threshold:
                    detections.append(([x1, y1, x2, y2], conf, int(cls)))

        # DeepSORT 跟踪
        tracks = deepsort.update_tracks(detections, frame=frame)

        # 遍历跟踪到的目标并绘制
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # 得到跟踪框
            x1, y1, x2, y2 = map(int, ltrb)

            # 绘制跟踪框和标签
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示实时视频
        cv2.imshow("Guardrail Detection & Tracking", frame)

        # 保存视频
        if args.save_video:
            out.write(frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    if args.save_video:
        out.release()
    cv2.destroyAllWindows()
    print("检测和跟踪完成，程序退出。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="护栏检测与跟踪")

    # 添加命令行参数
    parser.add_argument("--model_path", type=str, required=True, 
                        help="YOLO 模型路径")
    parser.add_argument("--camera_index", type=int, default=0, 
                        help="摄像头索引 (默认: 0)")
    parser.add_argument("--conf_threshold", type=float, default=0.3, 
                        help="检测置信度阈值 (默认: 0.3)")
    parser.add_argument("--save_video", action="store_true", 
                        help="是否保存检测与跟踪结果视频")
    parser.add_argument("--output_file", type=str, default="output_guardrail_camera_tracking.mp4", 
                        help="保存视频文件的路径")

    args = parser.parse_args()
    main(args)
