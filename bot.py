import io
import logging
import os
from collections import Counter
from functools import wraps

import aiogram
import cv2
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

from model.model import YoloOnnxDetection, YoloOnnxSegmentation
from user import User


TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_PATH = f"/bot/{TOKEN}"
WEBHOOK_URL = os.getenv("RENDER_EXTERNAL_URL") + WEBHOOK_PATH

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = os.getenv("PORT", default=8000)

bot = Bot(token=TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

users = {}


def auth(func):
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
    """Function to handle the start command and create new user
    :param message: message with user info
    """
    user_id = message.from_user.id
    if users.get(user_id) is None:
        language_code = message.from_user.language_code
        language = language_code if language_code in ["en", "ru"] else "en"
    else:
        language = users[user_id].language

    user = User(language)
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
    """Function to handle the help command. Prints a list of available bot commands
    :param message: message with user_id
    """
    user = users.get(message.from_user.id)
    language = user.language if user else message.from_user.language_code

    if language == "ru":
        text = (
            "Используйте команду /start для запуска бота \n"
            "Команда /settings позволяет сменить язык бота \n"
            "Отправьте боту изображение, и он покажет вам, какие объекты он увидел на картинке"
        )
    else:
        text = (
            "Please use /start command to start the bot \n"
            "You can use /settings command to change language \n"
            "Send an image to the bot and it will display the detected objects"
        )

    await message.answer(text)


@dp.message(Command("settings"))
@auth
async def setting_command(message: types.Message):
    """Function to handle the settings command and changing the bot language
    :param message: message with user_id
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
    """Function to handle the model command and changing the user's model
    :param message: message with user_id
    """
    if users[message.from_user.id].language == "ru":
        text = "Выберите модель \n\n"
    else:
        text = "Select the model \n\n"

    keyboard = [
        [
            types.InlineKeyboardButton(text="yolov8n", callback_data="yolov8n"),
            types.InlineKeyboardButton(text="yolov8n-seg", callback_data="yolov8n-seg"),
        ]
    ]
    await message.answer(
        text,
        parse_mode="markdown",
        reply_markup=types.InlineKeyboardMarkup(inline_keyboard=keyboard),
    )


async def model_forward(img: np.ndarray, user_id: int) -> tuple[np.ndarray, str]:
    language, model_name = usr_language[user_id], usr_model[user_id]
    net: YoloOnnxDetection | YoloOnnxSegmentation = models[model_name]

async def model_forward(img: np.ndarray, user: User) -> tuple[np.ndarray, str]:
    net: YoloOnnxDetection | YoloOnnxSegmentation = models[user.model]

    result = net(img, retina_masks=user.retina_masks)
    result_img = net.render(
        img,
        *result,
        hide_conf=False,
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
    """Detect objects on the image from the message"""
    user: User = users[message.from_user.id]

    file_io = io.BytesIO()
    await bot.download(message.photo[-1], destination=file_io)
    img = cv2.imdecode(np.frombuffer(file_io.read(), np.uint8), cv2.IMREAD_COLOR)

    result_img, text = await model_forward(img, user)

    _, img_encode = cv2.imencode(".jpg", result_img)
    result_img = types.BufferedInputFile(img_encode.tobytes(), "result_img")

    await message.answer_photo(result_img, caption=text)


@dp.callback_query(aiogram.F.func(lambda call: call.data.startswith("yolo")))
async def callback_model(call: types.CallbackQuery):
    """Callback function to handle the model buttons"""
    user_id = call.from_user.id
    new_model_name = call.data

    users[user_id].model = new_model_name

    if users[user_id].language == "ru":
        text = f"Сохранено!\nИспользуется модель {''.join(new_model_name.split('_'))}"
    else:
        text = f"Saved!\nUsing {''.join(new_model_name.split('_'))} model"

    await bot.send_message(user_id, text)


@dp.callback_query(aiogram.F.func(lambda call: call.data.startswith("language")))
async def callback_language(call: types.CallbackQuery):
    """Callback function to handle the language buttons"""
    new_language = call.data.split("_")[1]
    user_id = call.from_user.id

    if new_language == "ru":
        users[user_id].language = "ru"
        text = "Язык сохранён!"
    else:
        users[user_id].language = "en"
        text = "Language has been saved!"

    await bot.send_message(user_id, text)


async def on_startup() -> None:
    await bot.set_webhook(f"{WEBHOOK_URL}", drop_pending_updates=True)


async def on_shutdown():
    await bot.session.close()


def main():
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    # Create aiohttp.web.Application instance
    app = web.Application()

    # Create an instance of request handler
    webhook_requests_handler = SimpleRequestHandler(dispatcher=dp, bot=bot)

    # Register webhook handler on application
    webhook_requests_handler.register(app, path=WEBHOOK_PATH)

    # Mount dispatcher startup and shutdown hooks to aiohttp application
    setup_application(app, dp, bot=bot)

    # And finally start webserver
    web.run_app(app, host=WEBAPP_HOST, port=WEBAPP_PORT)


if __name__ == "__main__":
    model_list = [
        ("detection", "yolov8n"),
        ("segmentation", "yolov8n-seg"),
    ]

    model_types = {"segmentation": YoloOnnxSegmentation, "detection": YoloOnnxDetection}

    models = {
        model_name: model_types[model_type](
            checkpoint=f"checkpoints/{model_type}/{model_name}.onnx",
            input_size=(640, 640),
        )
        for model_type, model_name in model_list
    }
    # Warmup
    test_img = cv2.imread("images/bus.jpg", cv2.IMREAD_COLOR)
    for model in models.values():
        _ = model(test_img)

    main()
