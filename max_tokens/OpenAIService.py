from openai import OpenAI
from typing import List, Dict, Any, AsyncIterable, Optional, Union
from microsoft.tiktokenizer import create_by_model_name

class OpenAIService:
    def __init__(self):
        self.openai = OpenAI()
        self.tokenizers = {}
        self.IM_START = "<|im_start|>"
        self.IM_END = "<|im_end|>"
        self.IM_SEP = "<|im_sep|>"

    async def get_tokenizer(self, model_name: str):
        if model_name not in self.tokenizers:
            special_tokens = {
                self.IM_START: 100264,
                self.IM_END: 100265,
                self.IM_SEP: 100266,
            }
            tokenizer = await create_by_model_name(model_name, special_tokens)
            self.tokenizers[model_name] = tokenizer
        return self.tokenizers[model_name]

    async def count_tokens(self, messages: List[Dict[str, str]], model: str = 'gpt-4o') -> int:
        tokenizer = await self.get_tokenizer(model)

        formatted_content = ''
        for message in messages:
            formatted_content += f"{self.IM_START}{message['role']}{self.IM_SEP}{message.get('content', '')}{self.IM_END}"
        formatted_content += f"{self.IM_START}assistant{self.IM_SEP}"

        tokens = tokenizer.encode(formatted_content, [self.IM_START, self.IM_END, self.IM_SEP])
        return len(tokens)

    async def completion(self, config: Dict[str, Any]) -> Union[Dict[str, Any], AsyncIterable[Dict[str, Any]]]:
        messages = config['messages']
        model = config.get('model', 'gpt-4')
        stream = config.get('stream', False)
        json_mode = config.get('json_mode', False)
        max_tokens = config.get('max_tokens', 1024)

        try:
            chat_completion = await self.openai.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if json_mode else {"type": "text"}
            )
            
            return chat_completion
        except Exception as error:
            print("Error in OpenAI completion:", error)
            raise

    async def continuous_completion(self, config: Dict[str, Any]) -> str:
        messages = config['messages']
        model = config.get('model', 'gpt-4o')
        max_tokens = config.get('max_tokens', 1024)
        full_response = ""
        is_completed = False

        while not is_completed:
            completion = await self.completion({
                'messages': messages,
                'model': model,
                'max_tokens': max_tokens
            })
            
            choice = completion.choices[0]
            full_response += choice.message.content or ""

            if choice.finish_reason != "length":
                is_completed = True
            else:
                print("Continuing completion...")
                messages.extend([
                    {"role": "assistant", "content": choice.message.content},
                    {"role": "user", "content": "[system: Please continue your response to the user's question and finish when you're done from the very next character you were about to write, because you didn't finish your response last time. At the end, your response will be concatenated with the last completion.]"}
                ])

        return full_response 