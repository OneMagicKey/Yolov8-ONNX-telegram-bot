class Colors:
    # Copy from https://github.com/ultralytics/yolov5/blob/master/utils/plots.py
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=True):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


COCO_names_en = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

COCO_names_ru = (
    "человек",
    "велосипед",
    "автомобиль",
    "мотоцикл",
    "самолет",
    "автобус",
    "поезд",
    "грузовик",
    "лодка",
    "светофор",
    "пожарный гидрант",
    "знак остановки",
    "парковочный счетчик",
    "скамейка",
    "птица",
    "кошка",
    "собака",
    "лошадь",
    "овца",
    "корова",
    "слон",
    "медведь",
    "зебра",
    "жираф",
    "рюкзак",
    "зонтик",
    "сумка",
    "галстук",
    "чемодан",
    "фрисби",
    "лыжи",
    "сноуборд",
    "спортивный мяч",
    "воздушный змей",
    "бейсбольная бита",
    "бейсбольная перчатка",
    "скейтборд",
    "доска для серфинга",
    "теннисная ракетка",
    "бутылка",
    "бокал для вина",
    "чашка",
    "вилка",
    "нож",
    "ложка",
    "миска",
    "банан",
    "яблоко",
    "бутерброд",
    "апельсин",
    "брокколи",
    "морковь",
    "хот-дог",
    "пицца",
    "пончик",
    "торт",
    "стул",
    "диван",
    "комнатное растение",
    "кровать",
    "обеденный стол",
    "туалет",
    "телевизор",
    "ноутбук",
    "мышь",
    "пульт",
    "клавиатура",
    "мобильный телефон",
    "микроволновка",
    "духовка",
    "тостер",
    "раковина",
    "холодильник",
    "книга",
    "часы",
    "ваза",
    "ножницы",
    "плюшевый мишка",
    "фен",
    "зубная щетка",
)