import os
import asyncio
from openai import AsyncOpenAI

from LLMModel import LLMModel

class OpenAIAPI(LLMModel):

    __model_name: str = None
    __sampling_params: dict = None
    __client = None

    def __init__(self, model_name: str, sampling_params: dict):
        """**sampling_params**: dict with keys for vLLMModel sampling parameters (e.g., temperature)  """
        self.__model_name = model_name
        self.__sampling_params = sampling_params
        self.__client = AsyncOpenAI()

    async def request(self, input: str) -> str:

        api_response = await self.__client.responses.create(
            model = self.__model_name,
            input = input,
            **self.__sampling_params
        )

        return api_response.output_text

# module test
if __name__ == "__main__":

    async def main():

        llm_api = OpenAIAPI('gpt-5.1', {"temperature": 1})

        input_text = "What is the capital of France?"
        result = await llm_api.request(input_text)
        print(result)

    asyncio.run(main())