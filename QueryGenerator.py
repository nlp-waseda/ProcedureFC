import json
import asyncio
import regex
from typing import List
from LLMAgent import LLMAgent
from LLMModel import LLMModel

class QueryGenerator(LLMAgent):
    
    def __init__(self, llm_model: LLMModel, prompt_base: List[str], max_attempt_num: int):
        """prompt_base: [common, individual]"""
        super().__init__(llm_model, prompt_base, max_attempt_num)

    async def generateCommonQueries(self, user_question: str, llm_procedure: str) -> List[str]:

        """
        returrns a list of strings of queries
        """

        input = self._prompt_base[0].replace('<input-user-question>', user_question).replace('<input-procedure>', llm_procedure)

        finished = False
        for _ in range(self._max_attempt_num):

            print(f"==== {_+1}th attempt for CommonQuery ==== ") # debug

            response = await self._llm_model.request(input)

            matches = regex.findall(r"<json>(.*?)</json>", response, regex.DOTALL, overlapped=True) # <json>と</json>の間を抜き出す（リーズニングの過程タグが出力され、マッチ範囲がネストした場合、再帰的に検索を行い最後のマッチ結果を取り出す）
            if len(matches) > 0:
                json_text = matches[-1].strip()
                try:
                    response_dict = json.loads(json_text) # responseをjsonで読み込み、textを取り出す
                    if isinstance(response_dict, dict) and \
                        isinstance(response_dict.get("field"), str) and \
                            isinstance(response_dict.get("target"), str) and \
                                isinstance(response_dict.get("hypernym"), str):
                        finished = True
                        break # 全てのフォーマットチェックをクリアしたので、試行ループを抜ける
                    else:
                        pass
                except json.JSONDecodeError:
                    pass
            else:
                pass

        if finished == False:
            raise RuntimeError("Common Query Generation failed after maximum attempts.")
        
        queries = []
        # 分野 + 対象
        queries.append(f"{response_dict['field']} {response_dict['target']}")
        # 分野 + 上位概念
        queries.append(f"{response_dict['field']} {response_dict['hypernym']}")

        # youtubeとredditがページ数上限を圧迫しないよう除外
        queries = [q + " -site:youtube.com -site:reddit.com" for q in queries]

        return queries

    async def generateIndividualQueries(self, statements: List[str], c_queries: List[str]) -> List[dict]:

        """
        returns a list of dicts with keys: 'number', 'queries'
        """

        statements_txt = '\n'.join([f"{i+1}. {s}" for i, s in enumerate(statements)])
        c_queries_txt = ""
        for i in range(len(c_queries) - 1):
            c_queries_txt += f'"{c_queries[i]}", '
        c_queries_txt += f'"{c_queries[-1]}"'

        input = self._prompt_base[1].replace(
            '<input-statements>', statements_txt
        ).replace(
            '<input-c_queries>', c_queries_txt
        )

        response_dict = {}
        finished = False
        for _ in range(self._max_attempt_num):

            print(f"==== {_+1}th attempt for IndividualQuery ==== ") # debug

            response = await self._llm_model.request(input)

            matches = regex.findall(r"<json>(.*?)</json>", response, regex.DOTALL, overlapped=True) # <json>と</json>の間を抜き出す（リーズニングの過程タグが出力され、マッチ範囲がネストした場合、再帰的に検索を行い最後のマッチ結果を取り出す）

            if len(matches) > 0:
                json_text = matches[-1].strip()
                try:
                    response_dict = json.loads(json_text) # jsonで読み込み、dictに変換する
                    if isinstance(response_dict, list) and len(response_dict) == len(statements):
                        all_clear = True
                        for item in response_dict:
                            if isinstance(item, dict) and isinstance(item.get("number"), int) and isinstance(item.get("queries"), list) and len(item["queries"]) > 0:
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
            raise RuntimeError("Individual Query Generation failed after maximum attempts.")
        
        # クエリに除外するサイトを追加
        for item in response_dict:
            item['queries'] = [q + " -site:youtube.com -site:reddit.com" for q in item['queries']]

        return response_dict
    
# module test
if __name__ == "__main__":

    from OpenAIAPI import OpenAIAPI
    from vLLMModel import vLLMModel

    async def main():

        common_prompt = ""
        with open('prompts/prompt_query_common0.0.1.txt', encoding='utf-8') as f:
            common_prompt = f.read()
        individual_prompt = ""
        with open('prompts/prompt_query_individual0.0.0.txt', encoding='utf-8') as f:
            individual_prompt = f.read()
        prompt_base = [common_prompt, individual_prompt]
        #llm_api = OpenAIAPI('gpt-5.1', {"temperature": 0.5})
        #"""
        llm_api = vLLMModel("openai/gpt-oss-120b", {
            "temperature": 0.5,
            "top_p": 1,
            "top_k": -1,
            "max_tokens": 10000
        })
        #"""
        """
        llm_api = vLLMModel("Qwen/Qwen3-30B-A3B-Instruct-2507", {
                'temperature': 0.5,
                'top_p': 0.8,
                'top_k': 20,
                'max_tokens': 3000,
                'min_p': 0
            })
        """
        agent = QueryGenerator(llm_api, prompt_base, 5)
        
        user_question = "I have a question about the game Genshin Impact. Can you please explain step by step the steps required to obtain the dish \"Fine Tea, Full Moon\"?"
        procedure = "Sure — in **Genshin Impact**, the dish **\"Fine Tea, Full Moon\"** is part of a limited-time event recipe, so getting it involves a specific set of steps tied to that event.\nHere’s the step-by-step breakdown based on the most recent appearance of the dish:\n\n---\n\n### **Step-by-Step Guide to Obtain \"Fine Tea, Full Moon\"**\n\n**(as it appeared in the \"A Tea Party Beneath the Moon\" Mid-Autumn-style event)**\n\n1. **Check Event Availability**\n\n   * This dish is *not* in the permanent recipe pool. It’s typically available during certain seasonal events in Liyue (often around the Mid-Autumn Festival period in-game).\n   * Make sure you’ve updated the game and that the relevant event is active.\n\n2. **Participate in the Event Quests**\n\n   * Join the **event questline** where you interact with Liyue NPCs hosting a tea and food celebration.\n   * Progress through the quest until you unlock the cooking challenge or recipe rewards.\n\n3. **Unlock the Recipe**\n\n   * The recipe for \"Fine Tea, Full Moon\" is usually given as a **quest reward** or through the event shop.\n   * If it’s in the event shop, you’ll need to collect the event currency (by completing daily event challenges, mini-games, or combat domains).\n\n4. **Learn the Recipe**\n\n   * Open your **Inventory → Precious Items** tab.\n   * Select the \"Fine Tea, Full Moon\" recipe and click **Use** to learn it.\n\n5. **Gather Ingredients**\n   *(Ingredient list based on last event appearance)*\n\n   * **Tea Leaves** (event-exclusive ingredient, earned during the event)\n   * **Sugar** (purchasable from General Goods Shops)\n   * **Lotus Head** (found near Liyue lakes and rivers)\n   * **Milk** (purchasable from General Goods Shops)\n     *(Exact requirements may change per event re-run.)*\n\n6. **Cook the Dish**\n\n   * Go to any cooking station.\n   * Select **Fine Tea, Full Moon** from your learned recipes.\n   * Cook it manually for the first time to unlock auto-cook.\n\n---\n\n✅ **Note:** If the event has already ended, you can’t currently obtain \"Fine Tea, Full Moon\" until the developers re-run that event or add it to a permanent source.\n\n---\n\nIf you want, I can check **exact ingredient quantities and stats** for the most recent version of \"Fine Tea, Full Moon\" so you’ll know what it does in-game when cooked.\nDo you want me to include that?\n"
        statements = [
            "In the game Genshin Impact, the dish called \"Fine Tea, Full Moon\" is a limited-time event recipe that is tied to specific in-game events.",
            "In the game Genshin Impact, the dish \"Fine Tea, Full Moon\" is not part of the permanent recipe pool.",
            "In the game Genshin Impact, the dish \"Fine Tea, Full Moon\" is typically available only during certain seasonal events in Liyue, often around the in-game Mid-Autumn Festival period.",
            "In Genshin Impact, by updating the game client to the latest version, a player can ensure that time-limited events related to the dish \"Fine Tea, Full Moon\" are accessible.",
            "In Genshin Impact, by confirming that the relevant limited-time event for obtaining the dish \"Fine Tea, Full Moon\" is active, a player can verify that it is currently possible to acquire the recipe.",
            "In Genshin Impact, by joining the event questline in which Liyue NPCs host a tea and food celebration, a player can begin the process required to obtain the recipe for the dish \"Fine Tea, Full Moon.\"",
            "In Genshin Impact, by progressing through the event questline, a player can eventually unlock the associated cooking challenge or recipe rewards that may include the recipe for \"Fine Tea, Full Moon.\"",
            "In Genshin Impact, by obtaining the recipe for \"Fine Tea, Full Moon\" as a quest reward during the relevant event, a player can add this recipe to their inventory for later learning.",
            "In Genshin Impact, by purchasing the recipe for \"Fine Tea, Full Moon\" from the event shop when it is offered, a player can acquire the recipe using event currency."
        ]
        common_queries = await agent.generateCommonQueries(user_question, procedure)
        print("Common Queries:\n", common_queries)
        """
        common_queries = [
            "Genshin Impact Fine Tea, Full Moon -site:youtube.com -site:reddit.com",
            "Genshin Impact dish -site:youtube.com -site:reddit.com"
        ]
        """
        individual_queries = await agent.generateIndividualQueries(statements, common_queries)
        print("Individual Queries:\n\n")
        for i, iq in enumerate(individual_queries):
            print(f"Individual Query {i+1}: {iq}")

    asyncio.run(main())