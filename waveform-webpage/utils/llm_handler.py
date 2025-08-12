from openai import OpenAI
import json
from typing import Optional, Dict


def load_llm_client():
    with open("./llm_api_key") as file:
        api_info = json.load(file)
        api_url: str = api_info["api_url"]
        api_key: str = api_info["api_key"]
        model_name: str = api_info["model_name"]
    
    client = OpenAI(api_key=api_key, base_url=api_url)
    return client, model_name

def llm_query(prompt: str | list[Dict[str, str]], 
              system_prompt: Optional[str]=None,
              **kwargs
             ) -> Optional[str]:
    client, model_name = load_llm_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    elif isinstance(prompt, list):
        messages += prompt
    else:
        raise NotImplementedError(f"不支持的 prompt 类型：{type(prompt)}")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False,
            **kwargs
        )
        content: str = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"尝试获取回复过程中出错: {e}")
        return None
