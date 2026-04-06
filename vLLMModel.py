from typing import List
import asyncio
from transformers import AutoTokenizer
import vllm
from LLMModel import LLMModel

class vLLMModel(LLMModel):

    __tokenizer = None
    __llm = None
    __sp = None # SampingParams()メソッドを通したパラメータ
    __MAX_BUF = 100
    __request_buf: List[str] = []
    __responses: List[str] = None
    __generating: bool = False
    __observe_alive: bool = None
    __observe_loop_task = None
    
    def __init__(self, model_name: str, sampling_params: dict):
        """**sampling_params**: dict with keys for vLLMModel sampling parameters (e.g., temperature, top_p, top_k, max_tokens, min_p, etc.)  """
        super().__init__(model_name, sampling_params)
        self.__tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.__llm = vllm.LLM(self._model_name)
        self.__sp = vllm.SamplingParams(**self._sampling_params)
        self.__responses = [None] * self.__MAX_BUF
        self.__observe_alive = True

        self.__observe_loop_task = asyncio.create_task(self.__observe_loop()) # 監視ループの開始
    
    async def __observe_loop(self):

        while self.__observe_alive:
            
            if len(self.__request_buf) > 0:
                print(f"==== Request Buffer Length: {len(self.__request_buf)} ====") # debug
                self.__generating = True
                res = self.__llm.generate(self.__request_buf, self.__sp)
                self.__responses[:len(res)] = [r.outputs[0].text for r in res]
                self.__request_buf = []
                self.__generating = False
            
            await asyncio.sleep(10)

        print("I will finish.") # debug
    
    # override
    async def request(self, input: str) -> str:

        idx = None
        while True:

            # リクエストをバッファに追加
            if idx == None and self.__generating == False and len(self.__request_buf) < self.__MAX_BUF:
                idx = len(self.__request_buf)
                # print(f"Request: ({input[:50]}) added at index {idx}.") # debug
                self.__request_buf.append(input)
            
            # 応答が返ってきたら
            if idx != None and self.__responses[idx] != None:
                r = self.__responses[idx]
                self.__responses[idx] = None
                return r
            
            await asyncio.sleep(0.5)

    async def endObservation(self):
        self.__observe_alive = False
        await self.__observe_loop_task
        print("Observation ended.") # debug

# module test
if __name__ == "__main__":

    async def main():

        llm_model = vLLMModel("Qwen/Qwen3-30B-A3B-Instruct-2507", {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 3000,
            "min_p": 0
        })

        prompt_base = """Perform the following calculation and output the answer.
<Calculation>
<input-number> + 1000

Output your answer using the following format:
<Output Format>
Answer: (Input your answer)"""

        tasks = []
        for i in range(110):
            input = prompt_base.replace("<input-number>", str(i))
            tasks.append(asyncio.create_task(llm_model.request(input)))
        task_results = await asyncio.gather(*tasks)
        await llm_model.endObservation()
        for r in task_results:
            print(r)
    
    asyncio.run(main())