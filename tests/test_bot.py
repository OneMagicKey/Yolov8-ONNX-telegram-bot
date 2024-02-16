import os
import unittest

from src.utils import init_models


class BotTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dummy_token = "1234567890:ABCDEfgh-q1wERT2Yuiopasdfghjklzxc3v"
        os.environ["TELEGRAM_TOKEN"] = dummy_token

    def test_model_initialization(self):
        from bot import model_list

        self.assertIsInstance(init_models(model_list), dict)


if __name__ == "__main__":
    unittest.main()
