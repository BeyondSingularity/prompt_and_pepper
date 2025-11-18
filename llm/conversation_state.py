import os
import json

from utils import Singleton

class Storage(metaclass=Singleton):
    def __init__(self, file_path: str = "./data.json"):
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self) -> dict:
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return {}

    def save_data(self) -> None:
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=4)

    def get_conversation(self, user_id: str) -> list[dict[str, str]]:
        if user_id not in self.data:
            self.data[user_id] = []
        return self.data[user_id]
    
    def add_message(self, user_id: str, role: str, content: str) -> None:
        if user_id not in self.data:
            self.data[user_id] = []
        self.data[user_id].append({"role": role, "content": content})
        self.save_data()
    
    def clear_conversation(self, user_id: str) -> None:
        self.data[user_id] = []
        self.save_data()
