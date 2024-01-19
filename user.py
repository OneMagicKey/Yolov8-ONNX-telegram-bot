from dataclasses import dataclass
from typing import Literal


@dataclass
class User:
    language: Literal["en", "ru"] = "en"
    model: str = "yolov8n"
    color_scheme: Literal["equal", "random"] = "equal"
    retina_masks: bool = False
