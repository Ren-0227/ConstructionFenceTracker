import cv2

def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制边界框。
    :param frame: 图像帧
    :param bbox: 边界框坐标 (x1, y1, x2, y2)
    :param color: 边界框颜色
    :param thickness: 边界框线宽
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def draw_center_point(frame, center, color=(0, 0, 255), radius=5):
    """
    在图像上绘制中心点。
    :param frame: 图像帧
    :param center: 中心点坐标 (x, y)
    :param color: 中心点颜色
    :param radius: 中心点半径
    """
    cv2.circle(frame, center, radius, color, -1)