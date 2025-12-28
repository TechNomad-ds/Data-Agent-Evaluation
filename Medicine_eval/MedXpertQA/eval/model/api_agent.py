from openai import OpenAI
from typing import List
from model.base_agent import LLMAgent
import base64
import traceback
import time

def encode_image(image_path):
    if image_path:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        return "No image inputs"

class APIAgent(LLMAgent):
    def __init__(self, model_name, temperature=0) -> None:
        super().__init__(model_name, temperature)

        # --- 修改开始 ---
        # 替换为您自己的 API Key 和 Base URL
        self.api_key = "sk-OqIPE7A0rEMX8Rwt5NFrxB5TKAruSRGQVw7dUPRh78QpwGUi"
        self.base_url = "http://123.129.219.111:3000/v1" # 注意通常到 /v1 即可
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        # --- 修改结束 ---

        
        self.max_tokens = 8192

    def get_response(self, messages: List[dict]) -> str:
        if ("o3" in self.model_name) or ("o1" in self.model_name) or ("deepseek-reasoner" in self.model_name):
            messages = [m for m in messages if m["role"] != "system"]
            for _ in range(10):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        max_completion_tokens=self.max_tokens,
                        seed=0
                    )
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    time.sleep(1)
                    response = "No answer provided."
        else:
            for _ in range(10):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        temperature=self.temperature,
                        
                        max_tokens=self.max_tokens,
                        logprobs=True,
                        seed=0,
                    )
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    if "bad_response_status_code" in str(e):
                        print("Bad Response")
                        response = "No answer provided: bad_response."
                        break
                    elif "content_filter" in str(e):
                        print("Content Filter")
                        response = "No answer provided: content_filter."
                        break
                    else:
                        print(e)
                        print(traceback.format_exc())
                        time.sleep(1)
                        response = "No answer provided."
        try:
            log_probs = completion.choices[0].logprobs.content
            log_probs = [token_logprob.logprob for token_logprob in log_probs]
        except Exception as e:
            log_probs = []
        return response, log_probs

    def image_content(self, img_path: str) -> dict:
        img_path = img_path.strip()
        if img_path.startswith("http"):
            return {"type": "image_url", "image_url": {"url": img_path}}
        else:
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}}
