import os
from copy import deepcopy
from enum import Enum

import litellm
from dotenv import load_dotenv
from litellm import completion
from termcolor import colored

load_dotenv()

MESSAGE_TYPE = dict[str, str]
MESSAGES_TYPE = list[MESSAGE_TYPE] | None


litellm.vertex_project = os.environ.get("VERTEX_PROJECT", "talentmatch-dev")
litellm.vertex_location = os.environ.get("VERTEX_LOCATION", "us-central1")


class ModelName(str, Enum):
    GEMINI = "vertex_ai/gemini-pro"
    BISON = "vertex_ai/chat-bison-32k"


GEMINI_MODEL = ModelName.GEMINI


def chat_message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def system_message(content: str) -> dict[str, str]:
    return chat_message(role="system", content=content)


def user_message(content: str) -> dict[str, str]:
    return chat_message(role="user", content=content)


def assistant_message(content: str) -> dict[str, str]:
    return chat_message(role="assistant", content=content)


def oai_response(response) -> str:
    try:
        return response.choices[0].message.content
    except Exception:
        return response


def ask_llm(
    messages: MESSAGES_TYPE,
    model: str = GEMINI_MODEL,
    json_mode: bool | None = None,
) -> MESSAGES_TYPE:
    try:
        if json_mode is None and "json" in messages[-1]["content"].lower():
            response_format = {"type": "json_object"}
        else:
            response_format = None
        answer = completion(
            messages=deepcopy(messages),
            model=model,
            response_format=response_format,
        )
        answer = oai_response(answer)
        messages.append(assistant_message(content=answer))
        return messages
    except Exception as e:
        print(colored(f"\n\n{model} ERROR: {e}\n\n", "red"))
        return messages
