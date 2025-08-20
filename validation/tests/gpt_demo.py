import openai
from openai import OpenAI
import os
import base64
from dotenv import load_dotenv
load_dotenv()

def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')

def read_image(image_path):
    with open(image_path, 'rb') as f:
        image_content = f.read()
    return image_content

# o3
# client = openai.AzureOpenAI(
# api_key="BpmGhEEkbNaHKmAYNtVgJJqbCStCPerInlFyzq43VS3Da823H29CJQQJ99BFACHYHv6XJ3w3AAABACOGbUk5",
# api_version="2025-01-01-preview",
# azure_endpoint="https://teamx-gpt-o3.openai.azure.com"
# )
# resp = client.chat.completions.create(
#     model="TeamX-gpt-o3",
#     messages=[{"role": "system", "content": "You are a helpful assistant."},
#               {"role": "user", "content": "Who are you?"}]
# )

# client = openai.AzureOpenAI(
# api_key=os.getenv('AZURE_OPENAI_API_KEY'),
# api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
# azure_endpoint=os.getenv('AZURE_OPENAI_API_BASE')
# )

# # AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_API_BASE}/openai/deployments/${AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version=${AZURE_OPENAI_API_VERSION}

# resp = client.chat.completions.create(
#     model="TeamX-gpt-4o",
#     messages=[{"role": "system", "content": "You are a helpful assistant."},
#               {"role": "user", "content": "Who are you?"}],
#     temperature=0.7,
#     max_tokens= 256,
# )


# # computer use agent
# client = openai.AzureOpenAI(
#     azure_endpoint=os.getenv('AZURE_OPENAI_CUA_API_BASE'),
#     api_version=os.getenv('AZURE_OPENAI_CUA_API_VERSION'),
#     api_key=os.getenv('AZURE_OPENAI_CUA_API_KEY'),
# )

# resp = client.responses.create(
#     model="computer-use-preview",
#     tools=[{
#             "type": "computer_use_preview",
#             "display_width": 1920,
#             "display_height": 1080,
#             "environment": "linux"
#         }],
#     input=[{"role":"user",
#             "content":[
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/png;base64,{encode_image(read_image('E:/111.png'))}",
#                 },
#                 {
#                     "type": "input_text",
#                     "text": "帮我在 bing.com 搜 AI 最新新闻"
#                 }
#                 ]
#             }
#            ],
#     #input="Hello",
#     reasoning={
#                 "generate_summary": "concise",
#             },
#     truncation="auto"
# )

# o3-pro
# api_version="2025-04-01-preview"
# client = openai.AzureOpenAI(
#     api_key="EIIlM6gjMjanQnEcrg6btJEVDVGmiwjbSCzyamNRAkrTP6q060e3JQQJ99BFACHYHv6XJ3w3AAABACOGr193",
#     api_version=api_version,
#     azure_endpoint=f"https://teamx-gpt-o3-pro.openai.azure.com/openai/responses?openai/responses?api-version={api_version}"
# )

# resp = client.responses.create(
#     model = "TeamX-gpt-o3-pro",
#     input = "你好，o3-pro！"
# )

# o3 response()
# api_version="2025-04-01-preview"
# client = openai.AzureOpenAI(
#     api_key="BpmGhEEkbNaHKmAYNtVgJJqbCStCPerInlFyzq43VS3Da823H29CJQQJ99BFACHYHv6XJ3w3AAABACOGbUk5",
#     api_version=api_version,
#     azure_endpoint=f"https://teamx-gpt-o3.openai.azure.com/openai/responses?openai/responses?api-version={api_version}"
# )

# resp = client.responses.create(
#     model = "TeamX-gpt-o3",
#     input=[{"role":"user", "content": [{"type":"input_text", "text":"大阪世博会你最推荐的场馆是哪个？告诉我它的名字！"}]}],
#     reasoning={"summary": "detailed"}
# )

from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = 'https://sv-98272fdc-4f22-4a25-ba0e-7d51a37448d3-8000-x-defau-51acffd7d9.sproxy.hd-01.alayanew.com:22443/v1'
#openai_api_base = "http://localhost:8003/v1"
ckpt_path = "/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 目标图片路径
frame0 = "/capacity/userdata/vcfenxd75jiv/workshops/workshop-4caef9d8-9aba-46e5-aed0-46dcd955e590/uitars_training_sample16/36/2ad9387a-65d8-4e33-ad5b-7580065a27ca/537b97db-be75-43e8-a6e7-ea08a7f0a48c/image_0001.png"
frame0 = "results/pass@1_all_6env_66model_aligned_tmp0_08011310/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217_1754025337585/image_0000.png"
# frame1 = "results/pass@1_all_6env_66model_aligned_tmp0_08011310/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217_1754025337585/image_0001.png"
frame0 = "results/pass@1_all_6env_66model_aligned_tmp0_08011439/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217_1754030094367/image_0000.png"
# frame1 = "results/pass@1_all_6env_66model_aligned_tmp0_08011439/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217_1754030094367/image_0001.png"
# frame0 = "results/test_single/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217_1754060624831/image_0000.png"
# frame1 = "results/test_single/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217_1754060624831/image_0001.png"
# frame2 = "results/test_single/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217_1754060624831/image_0002.png"

frame0 = "/capacity/userdata/vcfenxd75jiv/workshops/workshop-4caef9d8-9aba-46e5-aed0-46dcd955e590/uitars_training_sample16/38/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217/24df5bbb-9698-4356-baff-6947550d7569/image_0001.png"

from pathlib import Path
with Path(frame0).open("rb") as f:
    obs_img_0 = f.read()
    
# with Path(frame1).open("rb") as f:
#     obs_img_1 = f.read()
    
# with Path(frame1).open("rb") as f:
#     obs_img_2 = f.read()

message = [
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.\n\n## Output Format\n```\nThought: ...\nAction: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1,y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1,y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1,y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n## Note\n- Use English in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.\n- My computer's password is 'password', feel free to use it when you need sudo rights.\n\n## User Instruction\nBrowse the natural products database.\n"
      },
      {
        "type": "image_url",
        "image_url": {"url": "data:image;base64," + encode_image(obs_img_0)}
      }
    ]
  }
]

# 用openai方式调用
# chat_response = client.chat.completions.create(
#         model=ckpt_path, #"ui_tars_1.5", #"tongui-32b",
#         messages=message,
#         temperature=0,
#         max_tokens=500,
#         frequency_penalty=1,
#         top_p=0.1,
#         seed=1
#         )
# resp = chat_response.choices[0].message.content

# 用post方式调用（和model_service_pool相同）
data = {"model": "ui_tars_1.5",
        "messages": message,
        "temperature":0,
        "frequency_penalty":1,
        "top_p":0.1,
        "seed":1
        }
import requests
url = f"{openai_api_base}/chat/completions"
response = requests.post(url, json=data)
print(response.json()["choices"][0]["message"]["content"])
#print(response.json())

#print(resp)
# print(resp.output[0].summary[0].text)
# print('\n')
# print(resp.output[1].content[0].text)