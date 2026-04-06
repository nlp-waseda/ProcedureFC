from typing import Callable, List
import asyncio

from OpenAIAPI import OpenAIAPI
from DecompositionAgent import DecompositionAgent
from QueryGenerator import QueryGenerator
from VerificationAgent import VerificationAgent
from SearchEngine import SearchEngine
from GoogleSE import GoogleSE

class FactcheckControl:

    __files_path: dict = None
    __llm_api: OpenAIAPI = None
    __decomposition_agent: DecompositionAgent = None
    __query_generator: QueryGenerator = None
    __verification_agent: VerificationAgent = None
    __search_engine: SearchEngine = None

    def __init__(self, files_path: dict):
        self.__files_path = files_path
        self.__llm_api = OpenAIAPI('gpt-5.1', {"temperature": 0.7})
        self.__search_engine = GoogleSE()
    
    def loadFactcheck(self):

        with open(self.__files_path['decomposition'], encoding='utf-8') as f:
            prompt_base = f.read()
            self.__decomposition_agent = DecompositionAgent(self.__llm_api, prompt_base)

        with open(self.__files_path['query'], encoding='utf-8') as f:
            prompt_base = f.read()
            self.__query_generator = QueryGenerator(self.__llm_api, prompt_base)

        with open(self.__files_path['verification'], encoding='utf-8') as f:
            prompt_base = f.read()
            self.__verification_agent = VerificationAgent(self.__llm_api, prompt_base)

    async def factcheck(self, user_question: str, procedure: str, callback_indicator: Callable[[int], None]) -> List[dict]:

        """
        callback_indicator(finished: int) ... finished = 0(decompose), 1(query), 2(search), 3(verify)  
        Returns a list of the dict of steps that were judged to be "False".  
        dict keys: "original_sentence", "reason"
        """

        # 分解タスク
        async def d() -> tuple[List[dict], List[dict]]:
            r  = await self.__decomposition_agent.decompose(procedure)
            callback_indicator(0)
            return r

        decompose_task = asyncio.create_task(d())

        # クエリ生成&Web検索タスク
        async def r() -> tuple[List[str], List[dict]]:
            queries = await self.__query_generator.generateQueries(user_question, procedure)
            callback_indicator(1)
            pages = []
            for q in queries:
                pages += await self.__search_engine.search(q)
            callback_indicator(2)
            return queries, pages
        
        retrieve_task = asyncio.create_task(r())

        # 分解タスクとクエリ生成&Web検索タスクを同時実行
        decompose_result, retrieve_result = await asyncio.gather(decompose_task, retrieve_task)
        steps, sup_info = decompose_result
        statements = steps + sup_info
        queries, pages = retrieve_result

        # 証拠の結合
        evidence = ''
        for i in pages:
            evidence += f"<title>\n{i['title']}\n<body>{i['text']}'\n\n'"

        # 検証タスクリスト
        verify_tasks = []
        for s in statements:
            verify_task = asyncio.create_task(
                self.__verification_agent.verify(evidence, str(s))
            )
            verify_tasks.append(verify_task)
        verify_results = await asyncio.gather(*verify_tasks)
        callback_indicator(3)

        # 検証結果のうち、falseのものだけを返す
        re = []
        for i, result in enumerate(verify_results):
            if result["result"] == False:
                s = statements[i]
                original = s.get("original_step_sentence") or s.get("text") or str(s)
                re.append({
                    "original_sentence": original,
                    "reason": result["reason"]
                })
        
        return re
    
# module test
if __name__ == "__main__":

    async def main():
        files_path = {
            'decomposition': 'prompts/prompt_decompose5.3.0.txt',
            'query': 'prompts/prompt_query0.1.0.txt',
            'verification': 'prompts/prompt_verify1.2.1.txt',
            'dataset': 'dataset/dataset0.3.0.json'
        }
        
        factcheck_control = FactcheckControl(files_path)
        factcheck_control.loadFactcheck()
        
        user_question = "I have a question about the game Genshin Impact. Can you please explain step by step the steps required to obtain the dish \"Fine Tea, Full Moon\"?"
        procedure = "Sure — in **Genshin Impact**, the dish **\"Fine Tea, Full Moon\"** is part of a limited-time event recipe, so getting it involves a specific set of steps tied to that event.\nHere’s the step-by-step breakdown based on the most recent appearance of the dish:\n\n---\n\n### **Step-by-Step Guide to Obtain \"Fine Tea, Full Moon\"**\n\n**(as it appeared in the \"A Tea Party Beneath the Moon\" Mid-Autumn-style event)**\n\n1. **Check Event Availability**\n\n   * This dish is *not* in the permanent recipe pool. It’s typically available during certain seasonal events in Liyue (often around the Mid-Autumn Festival period in-game).\n   * Make sure you’ve updated the game and that the relevant event is active.\n\n2. **Participate in the Event Quests**\n\n   * Join the **event questline** where you interact with Liyue NPCs hosting a tea and food celebration.\n   * Progress through the quest until you unlock the cooking challenge or recipe rewards.\n\n3. **Unlock the Recipe**\n\n   * The recipe for \"Fine Tea, Full Moon\" is usually given as a **quest reward** or through the event shop.\n   * If it’s in the event shop, you’ll need to collect the event currency (by completing daily event challenges, mini-games, or combat domains).\n\n4. **Learn the Recipe**\n\n   * Open your **Inventory → Precious Items** tab.\n   * Select the \"Fine Tea, Full Moon\" recipe and click **Use** to learn it.\n\n5. **Gather Ingredients**\n   *(Ingredient list based on last event appearance)*\n\n   * **Tea Leaves** (event-exclusive ingredient, earned during the event)\n   * **Sugar** (purchasable from General Goods Shops)\n   * **Lotus Head** (found near Liyue lakes and rivers)\n   * **Milk** (purchasable from General Goods Shops)\n     *(Exact requirements may change per event re-run.)*\n\n6. **Cook the Dish**\n\n   * Go to any cooking station.\n   * Select **Fine Tea, Full Moon** from your learned recipes.\n   * Cook it manually for the first time to unlock auto-cook.\n\n---\n\n✅ **Note:** If the event has already ended, you can’t currently obtain \"Fine Tea, Full Moon\" until the developers re-run that event or add it to a permanent source.\n\n---\n\nIf you want, I can check **exact ingredient quantities and stats** for the most recent version of \"Fine Tea, Full Moon\" so you’ll know what it does in-game when cooked.\nDo you want me to include that?\n"
        results = await factcheck_control.factcheck(user_question, procedure, lambda x: x)
        print(results)
    
    asyncio.run(main())