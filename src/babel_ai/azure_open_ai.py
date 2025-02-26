import os
from typing import Literal

from dotenv import load_dotenv
from openai import AzureOpenAI

# Define the Azure OpenAI endpoint and key

load_dotenv()

endpoint = os.getenv("AZURE_ENDPOINT")
api_key = os.getenv("AZURE_KEY")
api_version = "2024-10-01-preview"

# Create an instance of the AzureOpenAI client
CLIENT = AzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)


def azure_openai_request(
    messages: list,
    model: Literal[
        "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "o1-preview"
    ] = "gpt-4o-2024-08-06",
    temperature: float = 1.0,  # 0.0 to 1.0 higher = more creative
    frequency_penalty: float = 0.0,  # -2.0 to 2.0 higher = less repetition
    presence_penalty: float = 0.0,  # -2.0 to 2.0 higher = more diverse
    top_p: float = 1.0,  # 0.0 to 1.0 higher = more creative
    max_tokens: int = 1000,
) -> str:

    # Send the prompt to AzureOpenAI
    response = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Return the response
    return response.choices[0].message.content
