from babel_ai.azure_open_ai import azure_openai_request


class ChatbotConversation:
    def __init__(
        self,
        system_prompt: str,
        user_prompt: str = "",
    ):
        self.conversation = [
            {"role": "user", "content": system_prompt},
        ]
        if user_prompt:
            self.conversation.append({"role": "user", "content": user_prompt})

    def add_bot_message(self, bot_message: str):
        self.conversation.append({"role": "assistant", "content": bot_message})

    def add_user_message(self, user_message: str):
        self.conversation.append({"role": "user", "content": user_message})

    def get_conversation(self):
        return [
            message["role"] + ": " + message["content"]
            for message in self.conversation
        ]


def chatbot_conversation(
    bot_1_system_prompt: str,
    bot_2_system_prompt: str,
    starting_prompt: str = "",
    conversation_length: int = 10,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1000,
):
    bot_1 = ChatbotConversation(
        system_prompt=bot_1_system_prompt,
        user_prompt=starting_prompt,
    )
    bot_2 = ChatbotConversation(system_prompt=bot_2_system_prompt)

    for _ in range(conversation_length):
        bot_1_message = azure_openai_request(
            messages=bot_1.conversation,
            model="gpt-4o-2024-08-06",
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        bot_1.add_bot_message(bot_1_message)
        bot_2.add_user_message(bot_1_message)

        bot_2_message = azure_openai_request(
            messages=bot_2.conversation,
            model="gpt-4o-2024-08-06",
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        bot_2.add_bot_message(bot_2_message)
        bot_1.add_user_message(bot_2_message)

    return bot_1.get_conversation(), bot_2.get_conversation()
