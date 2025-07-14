
import os

import dotenv
from openai import AzureOpenAI

dotenv.load_dotenv()

def test_openai():
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_version = os.environ["AZURE_OPENAI_API_VERSION"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    model = os.environ["AZURE_OPENAI_MODEL"]
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
    messages = [
        {
            "role": "user",
            "content": "Hello"
        }
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages
    )

    print(resp.choices[0].message.content)

def test_print():
    print("hello world")