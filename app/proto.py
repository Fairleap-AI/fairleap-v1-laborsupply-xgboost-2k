import os
from dashscope import Generation
import dashscope
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
from dotenv import load_dotenv

load_dotenv()

def get_response(messages):
    print(f"MODEL_STUDIO_KEY : {os.getenv("MODEL_STUDIO_KEY")}")
    response = Generation.call(
        # If the environment variable is not configured, replace the following line with: api_key="sk-xxx",
        api_key=os.getenv("MODEL_STUDIO_KEY"),
        # Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        model="qwen-plus",
        messages=messages,
        result_format="message",
    )
    return response


messages = [
    {
        "role": "system",
        "content": """You are a clerk at the Bailian phone store, responsible for recommending phones to users. Phones have two parameters: screen size (including 6.1 inches, 6.5 inches, 6.7 inches) and resolution (including 2K, 4K).
        You can only ask the user one parameter at a time. If the user's information is incomplete, you need to ask a follow-up question to provide the missing parameter. If the parameter collection is complete, you should say: I have understood your purchase intention, please wait a moment.""",
    }
]

assistant_output = "Welcome to the Bailian phone store. What is the size of the phone do you need to buy?"
print(f"Model output: {assistant_output}\n")
while "I have understood your purchase intention" not in assistant_output:
    user_input = input("Please enter:")
    # Add user question information to the messages list
    messages.append({"role": "user", "content": user_input})
    assistant_output = get_response(messages).output.choices[0].message.content
    # Add the large model's response information to the messages list
    messages.append({"role": "assistant", "content": assistant_output})
    print(f"Model output: {assistant_output}")
    print("\n")