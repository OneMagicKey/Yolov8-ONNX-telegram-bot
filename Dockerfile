FROM python:3.10-slim-buster

ENV TELEGRAM_TOKEN="1234567890:ABCDEfgh-q1wERT2Yuiopasdfghjklzxc3v"

WORKDIR ./bot

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "src/bot.py"]
