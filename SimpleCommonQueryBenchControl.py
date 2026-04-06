import asyncio
from typing import List, Callable, Tuple
import json
import time
import os

from DecompositionAgent import DecompositionAgent
from QueryGenerator import QueryGenerator
from VerificationAgent import VerificationAgent
from LLMAgent import LLMAgent
from BenchmarkControl import BenchmarkControl
from DataType import *

class SimpleCommonQueryGenerator(LLMAgent):
    
    def __init__(self, llm_api, prompt_base: str):
        super().__init__(llm_api, prompt_base)
    
    async def generateQueries(self, user_question: str, llm_proceudre: str) -> List[str]:

        """
        returrns a list of strings of queries
        """

        input = self._prompt_base.replace('<input-user-question>', user_question).replace('<input-proceudre>', llm_proceudre)

        response = await self._llm_api.request(input)  

        # responseをjsonで読み込み、textを取り出す
        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError:
            return await self.generateQueries(user_question, llm_proceudre)  # フォーマットミスがあれば再請求

        queries = []
        queries.append(response_dict['text'])

        # youtubeとredditがページ数上限を圧迫しないよう除外
        queries = [q + " -site:youtube.com -site:reddit.com" for q in queries]

        return queries

class SimpleCommonQueryBenchControl(BenchmarkControl):

    def __init__(self, files_path: dict):
        super().__init__(files_path)

    # override
    def loadBench(self):

        with open(self._files_path['decomposition'], encoding='utf-8') as f:
            prompt_base = f.read()
            self._decomposition_agent = DecompositionAgent(self._llm_api, prompt_base)

        with open(self._files_path['simple_query'], encoding='utf-8') as f:
            prompt_base = f.read()
            self._query_generator = SimpleCommonQueryGenerator(self._llm_api, prompt_base)

        with open(self._files_path['verification'], encoding='utf-8') as f:
            prompt_base = f.read()
            self._verification_agent = VerificationAgent(self._llm_api, prompt_base)

        with open(self._files_path['dataset'], encoding='utf-8') as f:
            tasks = json.load(f)
            for task in tasks:
                self._task_entities.append(
                    TaskEntity(
                        ID=task['ID'],
                        user_question=task['user_question'],
                        procedure=task['procedure'],
                        label=task['label'],
                        supplementary_info=task['supplementary_info']
                    )
                )

    def __checkCorrectNum(self):
        
        for i in range(len(self._task_entities)):

            label = self._task_entities[i].label

            # ラベルごとにタスク数を記録
            if label == True: self._true_task_num += 1
            if label == False: self._false_task_num += 1

            # 各タスクの最終的な真実性を取得
            factuality = True
            step_results = self._task_results[i].verification_results
            for r in step_results:
                if r['result'] == False:
                    factuality = False
                    break
            
            # ラベルごとに正当数記録
            if factuality == label:
                if label == True: self._true_success_count += 1
                if label == False: self._false_success_count += 1
    
    # override
    async def runBench(self, on_task_done: Callable[[int, int], None]) -> Tuple[int]:
        
        """
        returns a tuple of (true_task_num, false_task_num, true_success_count, false_success_count)
        Results will be stored in self._task_results.
        on_task_done(finished_task_num: int, task_num: int) -> None
        """
        total = len(self._task_entities)
        self._task_results = [None] * total

        async def run_task(idx: int, task: TaskEntity) -> None:
            
            # 分解タスク
            decompose_task = asyncio.create_task(self._decomposition_agent.decompose(task.procedure))
            
            # クエリ生成&Web検索タスク
            async def r() -> tuple[str, List[dict]]:
                queries = await self._query_generator.generateQueries(task.user_question, task.procedure)
                pages = []
                for q in queries:
                    pages += await self._search_engine.search(q, 4)
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
                    self._verification_agent.verify(evidence, str(s))
                )
                verify_tasks.append(verify_task)
            verify_results = await asyncio.gather(*verify_tasks)

            self._task_results[idx] = TaskResult(task.ID, decompose_result, queries, pages, verify_results)

            on_task_done(idx, total)

        tasks = []
        for idx, task in enumerate(self._task_entities):
            tasks.append(asyncio.create_task(run_task(idx, task)))
        
        await asyncio.gather(*tasks)

        self.__checkCorrectNum()

        return (self._true_task_num, self._false_task_num, self._true_success_count, self._false_success_count)

    # override
    def saveResults(self) -> str:

        dir_path = 'results/SimpleQueryBench/' + time.strftime("%y%m%d-%H.%M.%S")
        os.makedirs(dir_path, exist_ok=True)


        # save benchmark summary
        with open(f"{dir_path}/benchmark_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"Date and Time:\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset:\t{self._files_path['dataset']}\n")
            f.write(f"Decomposition Prompt:\t{self._files_path['decomposition']}\n")
            f.write(f"Query Prompt:\t{self._files_path['simple_query']}\n")
            f.write(f"Verification Prompt:\t{self._files_path['verification']}\n")
            f.write(f"Accuracy:\t{(self._true_success_count+self._false_success_count)/(self._true_task_num+self._false_task_num):.2f}\n")
            f.write(f"True Accuracy:\t{self._true_success_count/self._true_task_num:.2f}\n")
            f.write(f"False Accuracy:\t{self._false_success_count/self._false_task_num:.2f}\n")

        # save decomposition results
        decomposition_results = [
            {'ID': result.ID, 'result': result.steps}
            for result in self._task_results
        ]
        with open(f"{dir_path}/decomposition_results.json", 'w', encoding='utf-8') as f:
            json.dump(decomposition_results, f, ensure_ascii=False, indent=2)

        # save query results
        query_results = [
            {'ID': result.ID, 'query': result.queries}
            for result in self._task_results
        ]
        with open(f"{dir_path}/query_results.json", 'w', encoding='utf-8') as f:
            json.dump(query_results, f, ensure_ascii=False, indent=2)

        # save search results
        search_results = [
            {'ID': result.ID, 'evidences': result.evidences}
            for result in self._task_results
        ]
        with open(f"{dir_path}/search_results.json", 'w', encoding='utf-8') as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)

        # save verification results
        verification_results = [
            {'ID': result.ID, 'verification_results': result.verification_results}
            for result in self._task_results
        ]
        with open(f"{dir_path}/verification_results.json", 'w', encoding='utf-8') as f:
            json.dump(verification_results, f, ensure_ascii=False, indent=2)

        return dir_path