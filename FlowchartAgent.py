import regex
import asyncio
import json

from LLMModel import LLMModel
from LLMAgent import LLMAgent

class FlowchartAgent(LLMAgent):

    def __init__(self, llm_model: LLMModel, prompt_base: str, max_attempt_num: int):
        super().__init__(llm_model, prompt_base, max_attempt_num)

    async def createFlowchart(self, procedure: str) -> str:

        """
        returns a flowchart in Mermaid format
        """

        input = self._prompt_base.replace('<input-procedure>', procedure)

        flowchart_text = ""
        finished = False
        for _ in range(self._max_attempt_num):

            print(f"==== {_+1}th attempt for flowchart ==== ") # debug

            response = await self._llm_model.request(input)

            matches = regex.findall(r"<flowchart>(.*?)</flowchart>", response, regex.DOTALL, overlapped=True) # <flowchart>と</flowchart>の間を抜き出す（リーズニングの過程タグが出力され、マッチ範囲がネストした場合、再帰的に検索を行い最後のマッチ結果を取り出す）
            if len(matches) > 0:
                flowchart_text = matches[-1].strip()
                if len(flowchart_text) > 30:
                    finished = True
                    break # 全てのフォーマットチェックをクリアしたので、試行ループを抜ける
                else:
                    pass
            else:
                pass

        if finished == False:
            raise RuntimeError("Flowchart failed after maximum attempts.")
        
        return flowchart_text

# module test
if __name__ == "__main__":

    from vLLMModel import vLLMModel
    from OpenAIAPI import OpenAIAPI

    async def main():

        prompt_base = ""
        with open('prompts/prompt_flowchart0.0.0.txt', encoding='utf-8') as f:
            prompt_base = f.read()

        """
        llm_model = vLLMModel('Qwen/Qwen3-30B-A3B-Instruct-2507', {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 3000,
            "min_p": 0
        })
        """
        #"""
        llm_model = vLLMModel("openai/gpt-oss-120b", {
            "temperature": 0.5,
            "top_p": 1,
            "top_k": -1,
            "max_tokens": 10000
        })
        #"""
        # llm_model = OpenAIAPI("gpt-5.1", {"temperature": 0.5})

        agent = FlowchartAgent(llm_model, prompt_base, 5)
        procedure = None
        with open('dataset/dataset0.5.1.json',encoding='utf-8') as f:
            data = json.load(f)
            procedure = data[0]['procedure']

        flowchart = await agent.createFlowchart(procedure)
        print(flowchart)

    asyncio.run(main())