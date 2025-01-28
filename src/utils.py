from dataclasses import dataclass
from typing import Literal

import numpy as np
from aiogram import types

from model.model import YoloOnnxDetection, YoloOnnxSegmentation


@dataclass
class ModelInfo:
    """
    A dataclass to represent information about a YOLO model.

    Attributes:
       type (str): The type of the YOLO model ('detection' or 'segmentation').
       name (str): The name of the YOLO model (e.g., 'yolov8s', 'yolov5s-seg').
       input_size (tuple[int, int]): The input size for the model, represented as a tuple of height and width. Default is (640, 640).
       conf (float): The confidence threshold for the model's predictions. Default is 0.25.
       iou (float): The IoU threshold for the non-maximum suppression. Default is 0.45.
       version (Literal[5, 8, 10, 11]): The version of the YOLO model, restricted to 5, 8, 10 or 11. Default is 8.
    """

    type: str
    name: str
    input_size: tuple[int, int] = (640, 640)
    conf: float = 0.25
    iou: float = 0.45
    version: Literal[5, 8, 10, 11] = 8


def init_models(
    model_list: list[ModelInfo],
) -> dict[str, YoloOnnxSegmentation | YoloOnnxDetection]:
    """
    Create and warm up Yolo models from the provided ModelInfo list
    """
    type2class = {"segmentation": YoloOnnxSegmentation, "detection": YoloOnnxDetection}

    bot_models = {
        model.name: type2class[model.type](
            checkpoint=f"src/checkpoints/{model.type}/{model.name}.onnx",
            input_size=model.input_size,
            conf=model.conf,
            iou=model.iou,
            version=model.version,
        )
        for model in model_list
    }

    # Warmup
    for model in bot_models.values():
        size = model.input_size
        test_img = np.ones((*size, 3), dtype=np.uint8)

        _ = model(test_img, raw=True)

    return bot_models


def create_keyboard_for_models(
    model_names: list[str], width: int = 2
) -> list[list[types.InlineKeyboardButton]]:
    """
    Create a keyboard of a given width with model names.

    :param model_names: list of model names
    :param width: width of the keyboard
    :return: telegram keyboard
    """
    assert width > 0, "keyboard width must be greater than 0"

    keyboard_models = [
        [
            types.InlineKeyboardButton(text=name, callback_data=name)
            for name in model_names[i : i + width]
        ]
        for i in range(0, len(model_names), width)
    ]

    return keyboard_models
