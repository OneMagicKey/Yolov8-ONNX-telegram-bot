from functools import wraps
from collections import Counter
import cv2
import io
import numpy as np
import os
import logging

import aiogram
from aiohttp import web
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from model.model import YoloOnnxSegmentation, YoloOnnxDetection

TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_PATH = f"/bot/{TOKEN}"
WEBHOOK_URL = os.getenv('RENDER_EXTERNAL_URL') + WEBHOOK_PATH

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = os.getenv("PORT", default=8000)

bot = Bot(token=TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

users = set()
usr_language = {}
usr_model = {}


def auth(func):
    @wraps(func)
    async def is_user_exists(message: types.Message):
        if message.from_user.id not in users:
            if message.from_user.language_code == 'ru':
                text = 'Сначала запустите бота командой\n/start и выберите язык'
            else:
                text = 'You should /start the bot and select the language'
            await message.answer(text)
        else:
            await func(message)

    return is_user_exists


@dp.message(Command('start'))
async def start_command(message: types.Message):
    """ Function to handle the start command and request bot language
    :param message: message with user info
    """
    user_id, language = message.from_user.id, message.from_user.language_code
    users.add(user_id)
    usr_language[user_id] = language if language in ['en', 'ru'] else 'en'

    if usr_language[user_id] == 'ru':
        text_greeting = 'Привет!\nОтправьте боту изображение, и он покажет вам, какие объекты он увидел на картинке'
        text_language = 'Выберите язык \n\n'
    else:
        text_greeting = "Welcome to the yolo bot!\nSend me an image and I will display the objects I've found"
        text_language = 'Select your language \n\n'

    usr_model[user_id] = 'yolov8n'
    await message.answer(text_greeting)
    await message.answer(text_language, parse_mode='markdown', reply_markup=types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text='Русский', callback_data='language_ru'),
         types.InlineKeyboardButton(text='English', callback_data='language_en')]
    ]))


@dp.message(Command('help'))
async def help_command(message: types.Message):
    """ Function to handle the help command. Prints a list of available bot commands
    :param message: message with user_id
    """
    if usr_language.get(message.from_user.id, message.from_user.language_code) == 'ru':
        text = (
            'Используйте команду /start для старта бота и выберете язык \n'
            'Команда /settings позволяет сменить язык бота \n'
            'Отправьте боту изображение, и он покажет вам, какие объекты он увидел на картинке'
        )
    else:
        text = (
            'Please use /start command to start the bot and select the language \n'
            'You can use /settings command to change language \n'
            'Send an image to the bot and it will display the detected objects'
        )
    await message.answer(text)


@dp.message(Command('settings'))
@auth
async def setting_command(message: types.Message):
    """ Function to handle the settings command and changing the bot language
    :param message: message with user_id
    """
    if usr_language[message.from_user.id] == 'ru':
        text = 'Выберите язык \n\n'
    else:
        text = 'Select your language \n\n'

    await message.answer(text, parse_mode='markdown', reply_markup=types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text='Русский', callback_data='language_ru'),
         types.InlineKeyboardButton(text='English', callback_data='language_en')]
    ]))


@dp.message(Command('model'))
@auth
async def model_command(message: types.Message):
    """ Function to handle the model command and changing the user's model
    :param message: message with user_id
    """
    user_id = message.from_user.id
    if usr_language[user_id] == 'ru':
        text = 'Выберите модель \n\n'
    else:
        text = 'Select the model \n\n'

    await message.answer(text, parse_mode='markdown', reply_markup=types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text='yolov8n', callback_data='yolov8n'),
         types.InlineKeyboardButton(text='yolov8n-seg', callback_data='yolov8n-seg')]
    ]))


async def model_forward(img: np.ndarray, user_id: int) -> tuple[np.ndarray, str]:
    language, model_name = usr_language[user_id], usr_model[user_id]
    net: YoloOnnxDetection | YoloOnnxSegmentation = models[model_name]

    result = net(img)
    result_img = net.render(img, *result, hide_conf=False, language=language)
    result_list = net.print_results(*result, language=language)

    counter = Counter(info.class_name for info in result_list)
    if not counter:
        text = 'Ничего не нашел' if usr_language[user_id] == 'ru' else 'No detections'
    else:
        text = '\n'.join(f'{count} {name}' for (name, count) in counter.items())

    return result_img, text


@dp.message(aiogram.F.photo)
@auth
async def process_image(message: types.Message):
    """ Detect objects on the image from the message"""
    user_id = message.from_user.id

    file_io = io.BytesIO()
    await bot.download(message.photo[-1], destination=file_io)
    img = cv2.imdecode(np.frombuffer(file_io.read(), np.uint8), cv2.IMREAD_COLOR)

    result_img, text = await model_forward(img, user_id)

    _, img_encode = cv2.imencode('.jpg', result_img)
    result_img = types.BufferedInputFile(img_encode.tobytes(), 'result_img')
    await message.answer_photo(result_img, caption=text)


@dp.callback_query(aiogram.F.func(lambda call: call.data.startswith('yolo')))
async def callback_model(call: types.CallbackQuery):
    """ Callback function to handle the model buttons"""
    model_name = call.data
    user_id = call.from_user.id
    usr_model[user_id] = model_name
    if usr_language[user_id] == 'ru':
        text = f"Сохранено!\nИспользуется модель {''.join(model_name.split('_'))}"
    else:
        text = f"Saved!\nUsing {''.join(model_name.split('_'))} model"

    await bot.send_message(user_id, text)


@dp.callback_query(aiogram.F.func(lambda call: call.data.startswith('language')))
async def callback_language(call: types.CallbackQuery):
    """ Callback function to handle the language buttons"""
    data = call.data.split('_')
    user_id = call.from_user.id
    if data[1] == 'ru':
        usr_language[user_id] = 'ru'
        text = 'Язык сохранён!'
    else:
        usr_language[user_id] = 'en'
        text = 'Language has been saved!'

    await bot.send_message(user_id, text)


async def on_startup() -> None:
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_webhook(f"{WEBHOOK_URL}")


async def on_shutdown():
    await bot.session.close()


def main():
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    # Create aiohttp.web.Application instance
    app = web.Application()

    # Create an instance of request handler
    webhook_requests_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot
    )
    # Register webhook handler on application
    webhook_requests_handler.register(app, path=WEBHOOK_PATH)

    # Mount dispatcher startup and shutdown hooks to aiohttp application
    setup_application(app, dp, bot=bot)

    # And finally start webserver
    web.run_app(app, host=WEBAPP_HOST, port=WEBAPP_PORT)


if __name__ == "__main__":
    model_list = [('detection', 'yolov8n'), ('segmentation', 'yolov8n-seg')]

    model_types = {'segmentation': YoloOnnxSegmentation, 'detection': YoloOnnxDetection}

    models = {
        model_name: model_types[model_type](f'checkpoints/{"/".join((model_type, model_name))}.onnx', (640, 640))
        for model_type, model_name in model_list
    }
    # Warmup
    test_img = cv2.imread('images/bus.jpg', cv2.IMREAD_COLOR)
    for model in models.values():
        _ = model(test_img)

    main()
