import abc

class LLMModel(abc.ABC):

    def __init__(self, model_name: str, sampling_params: dict):
        self._model_name = model_name
        self._sampling_params = sampling_params
    
    @abc.abstractmethod
    async def request(self, input: str) -> str:
        pass