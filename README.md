<h2 align="center">YOLOv8-ONNX-telegram-bot</h2>

<p align="center">
<img src="https://img.shields.io/github/languages/top/OneMagicKey/Yolov8-ONNX-telegram-bot">
<img src ="https://img.shields.io/github/repo-size/OneMagicKey/Yolov8-ONNX-telegram-bot">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

<img width="768" src="images/example.png">

Telegram bot for object detection and instance segmentation using YOLOv5/YOLOv8 implemented on python + openCV DNN + ONNX.

Additionally, the bot can be deployed on a [Render](https://render.com) cloud platform for free.

The webhook version of the running bot requires around 430MB of RAM and includes 2 
models: `yolov8n` and `yolov8n-seg`.  The pooling version includes 4 models: `yolov5n`, 
`yolov8n`, `yolov5n-seg`, and `yolov8n-seg`, and requires around 630MB of RAM.

## Features

- [x] ENG / RU languages
- [x] Object Detection / Instance Segmentation
- [x] Support webhooks to deploy as webservice ([webhooks-render](https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot/blob/webhooks-render/) branch)
- [x] Support pooling to run bot locally ([master](https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot/blob/master/) branch)


## Run bot locally
1) Create telegram bot with [BotFather](https://telegram.me/BotFather)
2) Clone the repo and install all the dependencies:
```
git clone https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot.git
cd Yolov8-ONNX-telegram-bot
pip install -r requirements.txt 
```
3) Add `TELEGRAM_TOKEN` provided by BotFather to the python environment
4) Run `bot.py`

## Deploy on [Render](https://render.com)
1) Create telegram bot with [BotFather](https://telegram.me/BotFather)
2) Create Render account
3) Go to `New` -> `Web Service` -> `Build and deploy from a Git repository`
4) Build web service with the following options:
   * Public Git repository - `https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot`
   * Branch - `webhooks-render`
   * Runtime - `Python 3`
   * Build Command - `pip install -r requirements.txt`
   * Start Command - `python bot.py`
   * Instance Type = `Free`
5) Add Environment Variables:
   * Name - `TELEGRAM_TOKEN`, value `{YOUR_TELEGRAM_TOKEN}`
   * Name - `PYTHON_VERSION`, value `3.10.13`
6) Deploy the app

Instances under the Render free plan will spin down after a period of inactivity, 
which is typically around 15 minutes. To prevent this, configure [cron-job](https://cron-job.org/) 
to send a POST request to the bot every 10 minutes, keeping it running permanently.

## References

1) [Ultralytics](https://github.com/ultralytics/ultralytics)
2) [yolov5_onnx_cv](https://github.com/brucefay1115/yolov5_onnx_cv)
