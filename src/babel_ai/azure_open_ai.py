import os

from dotenv import load_dotenv
from openai import AzureOpenAI

# Define the Azure OpenAI endpoint and key

load_dotenv()

endpoint = os.getenv("AZURE_ENDPOINT")
api_key = os.getenv("AZURE_KEY")
api_version = os.getenv("AZURE_VERSION")

# Create an instance of the AzureOpenAI client
CLIENT = AzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)


def azure_openai_request(
    messages: list,
    model: str = "gpt-4o",
    max_tokens: int = 150,
    temperature: float = 0.1,
    response_format: str = "json",
) -> str:

    # Send the prompt to AzureOpenAI
    response = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
    )

    # Return the response
    return response.choices[0].message.content
