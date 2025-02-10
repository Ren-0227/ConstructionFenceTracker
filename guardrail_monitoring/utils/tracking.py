from deep_sort_realtime.deepsort_tracker import DeepSort

def initialize_tracker(max_age=30):
    """
    初始化 DeepSORT 跟踪器。
    :param max_age: 跟踪目标的最大存活帧数
    :return: DeepSORT 跟踪器对象
    """
    return DeepSort(max_age=max_age)