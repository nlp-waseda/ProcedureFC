import json
import asyncio
from typing import List, Tuple
import regex

from LLMModel import LLMModel
from LLMAgent import LLMAgent

class DecompositionAgent(LLMAgent):

    def __init__(self, llm_model: LLMModel, prompt_base: str, max_attempt_num: int):
        super().__init__(llm_model, prompt_base, max_attempt_num) 
    
    async def decompose(self, procedure: str) -> List[dict]:

        """
        returns a list of dict with keys: 'number', 'sentence'
        """

        input = self._prompt_base.replace('<input-procedure>', procedure)

        response_dict = {}
        statements = []
        finished = False
        for _ in range(self._max_attempt_num):

            print(f"==== {_+1}th attempt for decomposition ==== ") # debug

            response = await self._llm_model.request(input)

            matches = regex.findall(r"<json>(.*?)</json>", response, regex.DOTALL, overlapped=True)  # <json>と</json>の間を抜き出す（リーズニングの過程タグが出力され、マッチ範囲がネストした場合、再帰的に検索を行い最後のマッチ結果を取り出す）
            if len(matches) > 0:
                json_text = matches[-1].strip()
                try:
                    response_dict = json.loads(json_text) # jsonで読み込み、dictに変換する
                    if isinstance(response_dict, dict) and isinstance(response_dict.get("statements"), list):
                        statements = response_dict["statements"]
                        all_clear = True
                        for s in statements:
                            if isinstance(s, dict) and isinstance(s.get("number"), int) and isinstance(s.get("sentence"), str):
                                pass
                            else:
                                all_clear = False
                                break
                        if all_clear:
                            finished = True
                            break # 全てのフォーマットチェックをクリアしたので、試行ループを抜ける
                    else:
                        pass
                except json.JSONDecodeError:
                    pass
            else:
                pass

        if finished == False:
            raise RuntimeError("Decomposition failed after maximum attempts.")

        return statements

# module test
if __name__ == "__main__":

    from vLLMModel import vLLMModel
    from OpenAIAPI import OpenAIAPI

    async def main():

        prompt_base = ""
        with open('prompts/prompt_decompose5.7.3.txt', encoding='utf-8') as f:
            prompt_base = f.read()

        # llm_model = OpenAIAPI("gpt-5.1", {"temperature": 0.5})
        """
        llm_model = vLLMModel('Qwen/Qwen3-30B-A3B-Instruct-2507', {
            "temperature": 0.5,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 3000,
            "min_p": 0
        })
        """

        llm_model = vLLMModel('openai/gpt-oss-120b', {
            "temperature": 0.5,
            "top_p": 1,
            "top_k": -1,
            "max_tokens": 10000
        })
        
        dummy_prompt_base = """Please output the following json text as is.
---
I will write the final result between <json> tags.
<json>
{
    "statement": "This is the final answer."
}
</json>"""
        agent = DecompositionAgent(llm_model, prompt_base, 5)
        procedure = "Sure — in **Genshin Impact**, the dish **\"Fine Tea, Full Moon\"** is part of a limited-time event recipe, so getting it involves a specific set of steps tied to that event.\nHere’s the step-by-step breakdown based on the most recent appearance of the dish:\n\n---\n\n### **Step-by-Step Guide to Obtain \"Fine Tea, Full Moon\"**\n\n**(as it appeared in the \"A Tea Party Beneath the Moon\" Mid-Autumn-style event)**\n\n1. **Check Event Availability**\n\n   * This dish is *not* in the permanent recipe pool. It’s typically available during certain seasonal events in Liyue (often around the Mid-Autumn Festival period in-game).\n   * Make sure you’ve updated the game and that the relevant event is active.\n\n2. **Participate in the Event Quests**\n\n   * Join the **event questline** where you interact with Liyue NPCs hosting a tea and food celebration.\n   * Progress through the quest until you unlock the cooking challenge or recipe rewards.\n\n3. **Unlock the Recipe**\n\n   * The recipe for \"Fine Tea, Full Moon\" is usually given as a **quest reward** or through the event shop.\n   * If it’s in the event shop, you’ll need to collect the event currency (by completing daily event challenges, mini-games, or combat domains).\n\n4. **Learn the Recipe**\n\n   * Open your **Inventory → Precious Items** tab.\n   * Select the \"Fine Tea, Full Moon\" recipe and click **Use** to learn it.\n\n5. **Gather Ingredients**\n   *(Ingredient list based on last event appearance)*\n\n   * **Tea Leaves** (event-exclusive ingredient, earned during the event)\n   * **Sugar** (purchasable from General Goods Shops)\n   * **Lotus Head** (found near Liyue lakes and rivers)\n   * **Milk** (purchasable from General Goods Shops)\n     *(Exact requirements may change per event re-run.)*\n\n6. **Cook the Dish**\n\n   * Go to any cooking station.\n   * Select **Fine Tea, Full Moon** from your learned recipes.\n   * Cook it manually for the first time to unlock auto-cook.\n\n---\n\n✅ **Note:** If the event has already ended, you can’t currently obtain \"Fine Tea, Full Moon\" until the developers re-run that event or add it to a permanent source.\n\n---\n\nIf you want, I can check **exact ingredient quantities and stats** for the most recent version of \"Fine Tea, Full Moon\" so you’ll know what it does in-game when cooked.\nDo you want me to include that?\n"

        try:
            statements = await agent.decompose(procedure)
        except RuntimeError as e:
            print(str(e))
            return
        
        for s in statements:
            print(s)
    
    asyncio.run(main())