import base64
import json

from openai import OpenAI

from mm_agents.prompts import COMPUTER_USE_DOUBAO


def test_call_uitars():
    client = OpenAI(
        api_key="empty",
        base_url="https://sv-1c80fcce-0ad0-4f4a-b139-63c0fbe9c925-8000-x-aps-o-fd30ebbcd8.sproxy.hd-01.alayanew.com:22443/v1"
    )
    print(client.models.list())
    with open("examples/1ce1ab97-fd16-4569-a2ff-b98ffa1c2ded/task_config.json") as f:
        config = json.load(f)

    with open("examples/1ce1ab97-fd16-4569-a2ff-b98ffa1c2ded/image_0002.png", "rb") as f:
        encoded_str = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": COMPUTER_USE_DOUBAO.format(
                        language="English",
                        instruction=config["instruction"]
                    )
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encoded_str}"}
                }
            ]
        }
    ]

    response = client.chat.completions.create(
                    model="ui_tars_1.5",
                    messages=messages,
                    frequency_penalty=1,
                    max_tokens=512,
                    temperature=1.0,
                    top_p=1.0,
                    extra_body={
                        "mm_processor_kwargs": {
                            "min_pixels": 16384,
                            "max_pixels": 16384*28*28,
                        },
                        "top_k": 50,
                    }
                )
    print(response.usage)
    print(response.choices[0].message.content)
