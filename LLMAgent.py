import abc
from typing import List
from LLMModel import LLMModel

class LLMAgent(abc.ABC):

    def __init__(self, llm_model: LLMModel, prompt_base: str | List[str], max_attempt_num: int):
        super().__init__()
        self._llm_model = llm_model
        self._prompt_base = prompt_base
        self._max_attempt_num = max_attempt_num