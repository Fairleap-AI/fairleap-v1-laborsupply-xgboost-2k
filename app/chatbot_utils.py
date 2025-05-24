import os
from dashscope import Generation
import dashscope
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
# from dotenv import load_dotenv

# load_dotenv()

def call_qwen(messages, model="qwen-plus"):
    # print(f"MODEL_STUDIO_KEY : {os.getenv("MODEL_STUDIO_KEY")}")
    response = Generation.call(
        # If the environment variable is not configured, replace the following line with: api_key="sk-xxx",
        api_key=os.getenv("MODEL_STUDIO_KEY"),
        # Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        model=model,
        messages=messages,
        result_format="message",
    )
    return response