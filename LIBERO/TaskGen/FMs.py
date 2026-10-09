from openai import OpenAI
import base64
import requests
import io
from PIL import Image
import time
from google import genai
from google.genai.types import HttpOptions, Part
import itertools

# Please don't leak your API keys to the world.
GEMINI_KEYS = [
    'AQ.Ab8RN6JMufZeYmCrdwOFon70Wh3uRasTwX-8UD75o3yyZrwWdQ'
    # "AQ.Ab8RN6I0G1MuvhKW9ohq5qaBSRVvt8Iw_jEM7FZXi_v1pIrzRw",
    # "AQ.Ab8RN6LwlTzdTBAFq9dDCQ1xZsklOlnAUUoZKx1lo0X88550xA", # Billing disabled
    # "AQ.Ab8RN6K47thqcnuWClhwAPaNB1PbcndOuzspGL-hcnmLuMfoow"
]
KEY_CYCLE = itertools.cycle(GEMINI_KEYS)

def get_next_gemini_key():
    """Rotate among available Gemini keys."""
    return next(KEY_CYCLE)

"""
GPT-5 API
"""

client = OpenAI(api_key="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")

def ask_gpt5_text(instruction: str, role_instruction: str = "You are a helpful assistant."):
    """
    调用 GPT-5 获取纯文本回答
    
    Note: GPT-5 does not support temperature parameter. The model uses its default sampling strategy.
    To encourage variance in outputs, modify the prompt to explicitly request varied solutions.
    
    :param instruction: 用户输入的问题或指令
    :param role_instruction: system role，定义模型身份/风格
    :return: GPT-5 的回答文本
    """
    for attempt in range(2):
        try:
            response = client.responses.create(
                model="gpt-5.1",
                input=[
                    {"role": "system", "content": role_instruction},
                    {"role": "user", "content": instruction}
                ]
            )
            return response.output_text
        except Exception as e:
            if attempt == 0 and ("429" in str(e) or "rate limit" in str(e).lower()):
                time.sleep(5)
                continue
            raise e

def ask_gpt5_image_v0(image_path: str, instruction: str, role_instruction: str = "You are a helpful assistant."):
    """
    调用 GPT-5，输入图片 + 指令
    :param image_path: 本地图片路径
    :param instruction: 用户输入的问题或指令
    :param role_instruction: system role，定义模型身份/风格
    :return: GPT-5 的回答文本
    """

    img = Image.open (image_path)
    img.thumbnail ((800, 800))
    buf = io.BytesIO ()
    img.save (buf, format="JPEG", quality=70)
    image_b64 = base64.b64encode (buf.getvalue ()).decode ("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_b64}"

    for attempt in range(2):
        try:
            response = client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system", "content": role_instruction},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": instruction},
                            {"type": "input_image", "image_url": image_data_url}
                        ]
                    }
                ]
            )
            return response.output_text
        except Exception as e:
            if attempt == 0 and ("429" in str(e) or "rate limit" in str(e).lower()):
                time.sleep(5)
                continue
            raise e

def ask_FM_image(image_path: str, instruction: str, role_instruction: str = "You are a helpful assistant.",model: str ="gpt"):
    # img = Image.open(image_path)
    # img.thumbnail((800, 800))
    # buf = io.BytesIO()
    # img.save(buf, format="JPEG", quality=80)
    # image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    # image_data_url = f"data:image/jpeg;base64,{image_b64}"

    img = Image.open (image_path)
    buf = io.BytesIO ()
    img.save (buf, format="JPEG", quality=100)  # 最高质量但仍然有损
    image_b64 = base64.b64encode (buf.getvalue ()).decode ("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_b64}"

    if model=="gpt":
        for attempt in range(2):
            try:
                response = client.responses.create(
                    model="gpt-5.1",
                    input=[
                        {"role": "system", "content": role_instruction},
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": instruction},
                                {"type": "input_image", "image_url": image_data_url}
                            ]
                        }
                    ],
                )
                return response.output_text
            except Exception as e:
                if attempt == 0 and ("429" in str(e) or "rate limit" in str(e).lower()):
                    time.sleep(5)
                    continue
                raise e
    elif model=="gemini":
        api_key = get_next_gemini_key()
        if not api_key:
            raise ValueError("Error: GEMINI_API_KEY not set.")

        # Model to use (defaulting to the one confirmed working, or use gemini-1.5-pro)
        # The user asked for "gemini-3-pro", but that may not exist publicly.
        # We'll use gemini-2.0-flash-exp (001) as it was verified.
        model_name = "gemini-3-pro-preview"

        # url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        # Note: The user's curl used aiplatform.googleapis.com. If the key is Vertex-specific,
        # we might need that endpoint. However, AI Studio keys usually use generativelanguage.
        # Let's try aiplatform first if the key starts with AQ (Vertex usually).

        # Heuristic: The curl command used aiplatform, so we use aiplatform.
        # url = f"https://aiplatform.googleapis.com/v1beta/publishers/google/models/{model_name}:generateContent?key={api_key}"
        url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model_name}:generateContent?key={api_key}"
        # url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        with open (image_path, "rb") as f:
            image_b64 = base64.b64encode (f.read ()).decode ("utf-8")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": role_instruction + "\n\n" + instruction},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": image_b64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0
            }
        }

        for attempt in range(2):
            try:
                response = requests.post (
                    url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps (payload)
                )
                if response.status_code == 429:
                    if attempt == 0:
                        time.sleep(5)
                        continue

                response.raise_for_status ()
                result = response.json ()
                # Parse response
                try:
                    return result ["candidates"] [0] ["content"] ["parts"] [0] ["text"]
                except (KeyError, IndexError):
                    return f"Error parsing Gemini response: {result}"

            except Exception as e:
                # Retry on 429 (Too Many Requests)
                if attempt == 0 and isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                    time.sleep(5)
                    continue

                print (f"Error calling Gemini: {e}")
                if 'response' in locals() and response.status_code != 200:
                    print (f"Response content: {response.text}")
                return f"Error: {e}"




import requests
import json
import os

def ask_gemini3_image(image_path: str, instruction: str, role_instruction: str = "You are a helpful assistant."):
    """
    Call Gemini via REST API (works on Python 3.8 w/o new SDK).
    """
    api_key = get_next_gemini_key()
    if not api_key:
        return "Error: GEMINI_API_KEY not set."

    # Model to use (defaulting to the one confirmed working, or use gemini-1.5-pro)
    # The user asked for "gemini-3-pro", but that may not exist publicly.
    # We'll use gemini-2.0-flash-exp (001) as it was verified.
    model_name = "gemini-3-pro-preview" 

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    # Note: The user's curl used aiplatform.googleapis.com. If the key is Vertex-specific,
    # we might need that endpoint. However, AI Studio keys usually use generativelanguage.
    # Let's try aiplatform first if the key starts with AQ (Vertex usually).
    
    # Heuristic: The curl command used aiplatform, so we use aiplatform.
    url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model_name}:generateContent?key={api_key}"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": role_instruction + "\n\n" + instruction},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": image_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0
        }
    }

    for attempt in range(2):
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            if response.status_code == 429:
                if attempt == 0:
                    time.sleep(5)
                    continue

            response.raise_for_status()
            result = response.json()
            # Parse response
            try:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                return f"Error parsing Gemini response: {result}"
            
        except Exception as e:
            if attempt == 0 and isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                time.sleep(5)
                continue

            print(f"Error calling Gemini: {e}")
            if 'response' in locals() and response.status_code != 200:
                print(f"Response content: {response.text}")
            return f"Error: {e}"

def ask_gemini3_text(instruction: str, role_instruction: str = "You are a helpful assistant."):
    """
    Call Gemini Text via REST API.
    """
    api_key = get_next_gemini_key()
    if not api_key:
        return "Error: GEMINI_API_KEY not set."

    model_name = "gemini-3-pro-preview"
    url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model_name}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": role_instruction + "\n\n" + instruction}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0
        }
    }

    for attempt in range(2):
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            if response.status_code == 429:
                if attempt == 0:
                    time.sleep(5)
                    continue

            response.raise_for_status()
            result = response.json()
            try:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                 return f"Error parsing Gemini response: {result}"

        except Exception as e:
            if attempt == 0 and isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                time.sleep(5)
                continue

            print(f"Error calling Gemini: {e}")
            if 'response' in locals() and response.status_code != 200:
                print(f"Response content: {response.text}")
            return f"Error: {e}"






"""
Qwen Local API
"""

def ask_qwen_text(instruction: str, role_instruction: str = "You are a helpful assistant.", server_url: str = "http://localhost:8000/generate"):
    """
    Call local Qwen3-VL server for text-only response.
    """
    try:
        response = requests.post(
            server_url,
            json={
                "prompt": instruction,
                "system_prompt": role_instruction
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"Error calling Qwen server: {e}")
        return f"Error: {e}"

def ask_qwen_image(image_path: str, instruction: str, role_instruction: str = "You are a helpful assistant.", server_url: str = "http://localhost:8000/generate"):
    """
    Call local Qwen3-VL server for image + text response.
    """
    try:
        # Read and encode image
        with open(image_path, "rb") as img_file:
            image_b64 = base64.b64encode(img_file.read()).decode('utf-8')

        response = requests.post(
            server_url,
            json={
                "prompt": instruction,
                "image_b64": image_b64,
                "system_prompt": role_instruction
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"Error calling Qwen server: {e}")
        return f"Error: {e}"

def timed_call(func, **kwargs):
    start = time.perf_counter()
    result = func(**kwargs)
    end = time.perf_counter()
    cost = end - start
    return result, cost

# 示例调用
if __name__ == "__main__":
    print (ask_gemini3_image ("test.jpg", "What is shown in this image? Just give me the name."))

    # result = ask_gemini_image_official (
    #     image_path="test.jpg",
    #     instruction="Tell me the name of the character.",
    #     api_key="AQ.Ab8RN6K47thqcnuWClhwAPaNB1PbcndOuzspGL-hcnmLuMfoow"
    # )
    #
    # print (result)
    #
    # # Ensure we have a test image
    # import os
    # path = "dataset/evaluation_images/sequence_collages/pref0_sequence_collage.png"
    # # Create a dummy test.jpg if not exists for backward compatibility of original test
    # if not os.path.exists("./test.jpg") and os.path.exists(path):
    #     # Just use the collage as test.jpg or use the collage path directly
    #     pass
    # else:
    #     # If neither exists, we might fail, but let's assume path exists based on file list
    #     pass
    #
    # instr = "Describe this image briefly."
    #
    # print("\n--- Testing Qwen (Local) ---")
    # if os.path.exists(path):
    #     answer_qwen, t_qwen = timed_call(
    #         ask_qwen_image,
    #         image_path=path,
    #         instruction=instr
    #     )
    #     print(f"Qwen time: {t_qwen:.3f} s")
    #     print("Qwen answer:", answer_qwen)
    # else:
    #     print(f"Test image not found at {path}, skipping Qwen image test.")
    #
    # print("\n--- Testing Qwen Text (Local) ---")
    # answer_qwen_text, t_qwen_text = timed_call(
    #     ask_qwen_text,
    #     instruction="What is the capital of France?"
    # )
    # print(f"Qwen Text time: {t_qwen_text:.3f} s")
    # print("Qwen Text answer:", answer_qwen_text)
    # # Original tests (commented out to avoid API costs/errors if key invalid or not needed)
    # # print("\n--- Testing GPT-5.1 ---")
    # # answer_51, t51 = timed_call(
    # #     ask_gpt5_image,
    # #     image_path=path,
    # #     instruction=instr
    # # )
    # # print(f"GPT-5.1 time: {t51:.3f} s")
    # # print("GPT-5.1 answer:", answer_51)
    #
