import yaml

from babel_ai.azure_open_ai import azure_openai_request


def transform_story(story: str, transform_prompt: str) -> str:
    """Transform a story using a given prompt."""

    # create a prompt message using the transform_prompt template
    try:
        prompt = transform_prompt.format(story=story)
    except KeyError:
        raise ValueError(
            "Transform prompt template must contain 'story' placeholder."
        )
    messages = [{"role": "system", "content": prompt}]

    # send story transformation prompt to Azure OpenAI
    response = azure_openai_request(
        messages=messages,
        max_tokens=2 * len(story),
    )
    # TODO: Create response format that allows to
    #       extract the reversed story easily.

    return response


def reverse_story_moral(story: str) -> str:
    # Load the reversal prompt from the YAML file
    with open(
        "/Users/mienhardt/Programming/babel_ai/data/reversal_prompt.yml", "r"
    ) as file:
        reversal_prompts = yaml.safe_load(file)

    reversal_prompt = reversal_prompts["prompt_1"]

    # Transform the story using the reversal prompt
    reversed_story = transform_story(
        story=story, transform_prompt=reversal_prompt
    )

    return reversed_story
