from dataclasses import dataclass
from typing import Literal

import cv2
from aiogram import types

from model.model import YoloOnnxDetection, YoloOnnxSegmentation


@dataclass
class ModelInfo:
    """
    Dataclass for storing information about a model before initialisation.
    """

    type: str
    name: str
    input_size: tuple[int, int]
    version: Literal[5, 8]


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
            version=model.version,
        )
        for model in model_list
    }

    # Warmup
    test_img = cv2.imread("src/images/bus.jpg", cv2.IMREAD_COLOR)
    for model in bot_models.values():
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
