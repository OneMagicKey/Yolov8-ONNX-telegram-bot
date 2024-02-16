import asyncio
import io
import logging
import os
from collections import Counter
from dataclasses import dataclass
from functools import wraps
from typing import Literal

import aiogram
import cv2
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command

from model.model import YoloOnnxDetection, YoloOnnxSegmentation
from utils import ModelInfo, create_keyboard_for_models, init_models


@dataclass
class User:
    model: str
    language: Literal["en", "ru"] = "en"
    color_scheme: Literal["equal", "random"] = "equal"
    retina_masks: bool = False


TOKEN = os.getenv("TELEGRAM_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

users: dict[int, User] = {}


def auth(func):
    """
    Authentication decorator.

    Users must /start the bot before they can use it.
    """

    @wraps(func)
    async def is_user_exists(message: types.Message):
        if message.from_user.id not in users:
            if message.from_user.language_code == "ru":
                text = "Сначала запустите бота командой\n/start"
            else:
                text = "Please /start the bot first"
            await message.answer(text)
        else:
            await func(message)

    return is_user_exists


@dp.message(Command("start"))
async def start_command(message: types.Message):
    """
    Function to handle the start command and create a new user.

    :param message: user message
    """
    user_id = message.from_user.id
    if users.get(user_id) is None:
        language_code = message.from_user.language_code
        language = language_code if language_code in ["en", "ru"] else "en"
    else:
        language = users[user_id].language

    user = User(model_list[0].name, language)  # use the first model in the list as default  # fmt: skip
    users[user_id] = user

    if user.language == "ru":
        text_greeting = (
            "Привет!\nОтправьте боту изображение, и он покажет вам, "
            "какие объекты он увидел на картинке"
        )
    else:
        text_greeting = (
            "Welcome to the yolo bot!\nSend me an image and "
            "I will display the objects I've found"
        )

    await message.answer(text_greeting)


@dp.message(Command("help"))
async def help_command(message: types.Message):
    """
    Function to handle the help command. Prints a list of available bot commands.

    :param message: user message
    """
    user = users.get(message.from_user.id)
    language = user.language if user else message.from_user.language_code

    if language == "ru":
        text = (
            "Используйте команду /start для запуска бота\n"
            "\nКоманда /language позволяет выбрать язык бота, команда /model изменяет "
            "архитектуру модели. С помощью команды /color_scheme можно изменять "
            "цветовую схему найденных объектов\n"
            "\nОтправьте боту изображение, и он покажет вам, какие объекты он увидел "
            "на картинке"
        )
    else:
        text = (
            "Please use /start command to start the bot\n"
            "\nUse the /language command to set the language preferences and the "
            "/model command to select the architecture of the model. The /color_scheme "
            "command allows you to change the colour scheme of detected objects\n"
            "\nSend an image to the bot and it will display the detected objects"
        )

    await message.answer(text)


@dp.message(Command("language"))
@auth
async def language_command(message: types.Message):
    """
    Function to change the bot language.

    :param message: user message
    """
    if users[message.from_user.id].language == "ru":
        text = "Выберите язык \n\n"
    else:
        text = "Select your language \n\n"

    keyboard = [
        [
            types.InlineKeyboardButton(text="Русский", callback_data="language_ru"),
            types.InlineKeyboardButton(text="English", callback_data="language_en"),
        ]
    ]
    await message.answer(
        text,
        parse_mode="markdown",
        reply_markup=types.InlineKeyboardMarkup(inline_keyboard=keyboard),
    )


@dp.message(Command("model"))
@auth
async def model_command(message: types.Message):
    """
    Function to handle the model command and changing the user's model.

    :param message: user message
    """
    if users[message.from_user.id].language == "ru":
        text = "Выберите модель \n\n"
    else:
        text = "Select a model \n\n"

    await message.answer(
        text,
        parse_mode="markdown",
        reply_markup=types.InlineKeyboardMarkup(inline_keyboard=model_command_keyboard),
    )


@dp.message(Command("color_scheme"))
@auth
async def color_scheme_command(message: types.Message):
    """
    Function to change a colour scheme for detected objects.

    :param message: user message
    """
    if users[message.from_user.id].language == "ru":
        text = "Выберите цветовую схему \n\n"
    else:
        text = "Select a color scheme \n\n"

    keyboard = [
        [
            types.InlineKeyboardButton(text="equal", callback_data="color_equal"),
            types.InlineKeyboardButton(text="random", callback_data="color_random"),
        ],
    ]
    await message.answer(
        text,
        parse_mode="markdown",
        reply_markup=types.InlineKeyboardMarkup(inline_keyboard=keyboard),
    )


@dp.message(Command("retina_masks"))
@auth
async def retina_masks_command(message: types.Message):
    """
    Function to enable high-quality masks for instance segmentation.

    :param message: user message
    """
    if users[message.from_user.id].language == "ru":
        text = "Использовать маски в высоком разрешении при сегментации? \n\n"
    else:
        text = "Enable high-quality masks? \n\n"

    keyboard = [
        [
            types.InlineKeyboardButton(text="yes", callback_data="retina_on"),
            types.InlineKeyboardButton(text="no", callback_data="retina_off"),
        ],
    ]
    await message.answer(
        text,
        parse_mode="markdown",
        reply_markup=types.InlineKeyboardMarkup(inline_keyboard=keyboard),
    )


async def model_forward(img: np.ndarray, user: User) -> tuple[np.ndarray, str]:
    """
    Perform a forward pass through the model and render the resulting image

    :param img: image from the user's message
    :param user: user parameters
    :return: input image with bounding boxes and (optional) masks drawn on top
    """
    net: YoloOnnxDetection | YoloOnnxSegmentation = models[user.model]

    result = net(img, retina_masks=user.retina_masks)
    result_img = net.render(
        img,
        *result,
        language=user.language,
        color_scheme=user.color_scheme,
    )
    result_list = net.print_results(*result, language=user.language)

    counter = Counter(info.class_name for info in result_list)
    if not counter:
        text = "Ничего не нашел" if user.language == "ru" else "No detections"
    else:
        text = "\n".join(f"{count} {name}" for (name, count) in counter.items())

    return result_img, text


@dp.message(aiogram.F.photo)
@auth
async def process_image(message: types.Message):
    """
    Detect objects in the image from the message.

    :param message: user message with an attached image
    """
    user: User = users[message.from_user.id]

    file_io = await message.bot.download(file=message.photo[-1])
    img = cv2.imdecode(np.frombuffer(file_io.read(), np.uint8), cv2.IMREAD_COLOR)

    result_img, text = await model_forward(img, user)

    _, img_encode = cv2.imencode(".jpg", result_img)
    result_img = types.BufferedInputFile(img_encode.tobytes(), "result_img")

    await message.answer_photo(result_img, caption=text)


@dp.callback_query(aiogram.F.func(lambda call: call.data.startswith("yolo")))
async def callback_model(call: types.CallbackQuery):
    """
    Callback function to handle model keyboard buttons.
    """
    user_id = call.from_user.id
    new_model_name = call.data

    users[user_id].model = new_model_name

    if users[user_id].language == "ru":
        text = f"Сохранено!\nИспользуется модель {new_model_name}"
    else:
        text = f"Saved!\nThe {new_model_name} model is used"

    await bot.send_message(user_id, text)


@dp.callback_query(aiogram.F.func(lambda call: call.data.startswith("language")))
async def callback_language(call: types.CallbackQuery):
    """
    Callback function to handle language keyboard buttons.
    """
    new_language = call.data.split("_")[1]
    user_id = call.from_user.id

    if new_language == "ru":
        users[user_id].language = "ru"
        text = "Язык сохранён!"
    else:
        users[user_id].language = "en"
        text = "Language has been saved!"

    await bot.send_message(user_id, text)


@dp.callback_query(aiogram.F.func(lambda call: call.data.startswith("color")))
async def callback_color_scheme(call: types.CallbackQuery):
    """
    Callback function to handle color_scheme keyboard buttons.
    """
    user_id = call.from_user.id
    new_color_scheme = call.data.split("_")[1]

    users[user_id].color_scheme = new_color_scheme

    if users[user_id].language == "ru":
        text = f"Сохранено!\nИспользуется цветовая схема {new_color_scheme}"
    else:
        text = f"Saved!\nThe {new_color_scheme} color scheme is used"

    await bot.send_message(user_id, text)


@dp.callback_query(aiogram.F.func(lambda call: call.data.startswith("retina")))
async def callback_retina_masks(call: types.CallbackQuery):
    """
    Callback function to handle retina_masks keyboard buttons.
    """
    user_id = call.from_user.id
    retina_masks_flag = call.data.split("_")[1] == "on"

    users[user_id].retina_masks = retina_masks_flag

    if users[user_id].language == "ru":
        mask_text = "высоком разрешении" if retina_masks_flag else "обычном разрешении"
        text = f"Сохранено!\nИспользуются маски в {mask_text}"
    else:
        mask_text = "Retina" if retina_masks_flag else "Default"
        text = f"Saved!\n{mask_text} masks are used"

    await bot.send_message(user_id, text)


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_my_commands(
        [
            types.BotCommand(command="start", description="start the bot"),
            types.BotCommand(command="help", description="list of available commands"),
            types.BotCommand(
                command="language", description="set language preferences"
            ),
            types.BotCommand(command="model", description="select a model"),
            types.BotCommand(
                command="color_scheme",
                description="select a color scheme for detected objects",
            ),
            types.BotCommand(
                command="retina_masks",
                description="enable high-quality segmentation masks",
            ),
        ]
    )
    await dp.start_polling(bot)


model_list = [
    ModelInfo("detection", "yolov8s", (640, 640), 8),
    ModelInfo("detection", "yolov5s", (640, 640), 5),
    ModelInfo("segmentation", "yolov8s-seg", (640, 640), 8),
    ModelInfo("segmentation", "yolov5s-seg", (640, 640), 5),
]

if __name__ == "__main__":
    models = init_models(model_list)
    model_command_keyboard = create_keyboard_for_models(list(models.keys()))

    asyncio.run(main())
