import asyncio
from typing import List, Callable, Tuple
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import json
import regex
import time
import os

from BenchmarkControl import BenchmarkControl
from DataType import *

class OpenAIBenchControl(BenchmarkControl):

    __prompt_base: str = ""
    __task_results: List[dict] = [] # 結果の形式が違うため、TaskResultクラスは使えない

    def __init__(self, files_path: dict, model_info: dict):
        """
        **files_path**: dict with keys 'flowchart', 'decomposition', 'query', 'verification', 'verification_FC', 'dataset'  
        **model_info**: dict with keys 'model_name', 'sampling_params'  
        **sampling_params**: dict with keys for OpenAI API sampling parameters
        """
        super().__init__(files_path, model_info)
        self._openai_client = AsyncOpenAI()
    
    async def __requestOpenAI(self, prompt: str, max_attempt_num: int = 5) -> dict:
        
        """
        Returns dict with keys 'reason': str, 'result': bool, 'reasoning_results': List[str], 'search_results': dict.  
        Keys of search_results are queries, values are list of sources.
        """

        res = {}
        finished = False
        for i in range(max_attempt_num):

            print(f"==== {i+1}th attempt for OpenAI request ==== ") # debug

            response = await self._openai_client.responses.create(
                model = self._model_info["model_name"],
                input = prompt,
                **self._model_info["sampling_params"]
            )

            outputs = response.output
            reasoning_summaries: List[str] = []
            search_results: dict = {}
            none_count = 0 # クエリがNoneの検索が複数行われた時用
            for o in outputs:
                
                # 推論の要約の取得
                if getattr(o, "type", None) == "reasoning":
                    if len(getattr(o, "summary", [])) > 0:
                        for s in o.summary:
                            reasoning_summaries.append(s.text)
                
                # Web検索の結果の取得
                if getattr(o, "type", None) == "web_search_call":

                    # queryキーの取得のみデフォルト値を変えていることについて、ソースを取得していてもqueryはNoneになっている場合がままあるようなので、Noneでは弾かず、queryというキー自体が存在しないときのみ弾いている
                    if getattr(o, "action", None) != None and \
                        getattr(o.action, "query", "<no query>") != "<no query>" and \
                        isinstance(getattr(o.action, "sources", None), list):

                        sources = [getattr(item, "url") for item in o.action.sources]

                        if o.action.query == None: # クエリがNoneになっている場合は、None0, None1,...とキーをつける
                            search_results[f"None{none_count}"] = sources    
                            none_count += 1
                        else:
                            search_results[o.action.query] = sources

            res["search_results"] = search_results
            res["reasoning_results"] = reasoning_summaries

            # 応答の文章の取得
            response_text = response.output_text

            matches = regex.findall(r"<json>(.*?)</json>", response_text, regex.DOTALL, overlapped=True)  # <json>と</json>の間を抜き出す（リーズニングの過程タグが出力され、マッチ範囲がネストした場合、再帰的に検索を行い最後のマッチ結果を取り出す）
            if len(matches) > 0:
                json_text = matches[-1].strip()
                try:
                    response_dict = json.loads(json_text) # jsonで読み込み、dictに変換する
                    if isinstance(response_dict, dict) and isinstance(response_dict.get("reason"), str) and isinstance(response_dict.get("result"), bool):
                        res["reason"] = response_dict["reason"]
                        res["result"] = response_dict["result"]
                        finished = True
                        break # 全てのフォーマットチェックをクリアしたので、試行ループを抜ける
                    else:
                        pass
                except json.JSONDecodeError:
                    pass
            else:
                pass

        if finished == False:
            raise RuntimeError("OpenAI request failed after maximum attempts.")

        return res

    def __checkCorrectNum(self):
        
        for i in range(len(self._task_entities)):

            label = self._task_entities[i].label

            # ラベルごとにタスク数を記録
            if label == True: self._true_task_num += 1
            if label == False: self._false_task_num += 1

            # 各タスクの最終的な真実性を取得
            factuality = None
            if isinstance(self.__task_results[i]['result'], bool):
                factuality = self.__task_results[i]['result']
            else:
                continue
            
            # ラベルごとに正当数記録
            if factuality == label:
                if label == True: self._true_success_count += 1
                if label == False: self._false_success_count += 1
    
    # override
    def loadBench(self):

        with open(self._files_path['simple_llm_bench'], encoding='utf-8') as f:
            self.__prompt_base = f.read()

        with open(self._files_path['dataset'], encoding='utf-8') as f:
            tasks = json.load(f)
            if 'dataset_range' in self._files_path: # コマンドライン引数でデータセットの範囲が指定されている場合
                tasks = tasks[self._files_path['dataset_range'][0]:self._files_path['dataset_range'][1]+1]
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
    
    # override
    async def runBench(self, on_task_done: Callable[[int, int], None]) -> Tuple[int]:
        
        """
        returns a tuple of (true_task_num, false_task_num, true_success_count, false_success_count)
        Results will be stored in self._task_results.
        on_task_done(finished_task_num: int, task_num: int) -> None
        """

        total = len(self._task_entities)
        self.__task_results = [None] * total # self._task_resultsでないことに注意

        open("bench_log.jsonl", "w", encoding='utf-8').close()  # ログファイルを初期化

        async def run_task(idx: int, task: TaskEntity) -> None:

            prompt = self.__prompt_base.replace(
                '<input-procedure>', task.procedure).replace(
                '<input-question>', task.user_question
            )
            
            try:
                task_result = await self.__requestOpenAI(prompt)
            except RuntimeError:
                raise

            self.__task_results[idx] = task_result
        
        tasks = []
        for idx, task in enumerate(self._task_entities):
            tasks.append(asyncio.create_task(run_task(idx, task)))

        exceptions = await asyncio.gather(*tasks, return_exceptions=True)
        for i, e in enumerate(exceptions):
            if isinstance(e, Exception):
                self.__task_results[i] = {
                    "reason": None, "result": str(e), "reasoning_results": None, "search_results": None
                }

        self.__checkCorrectNum()

        return (self._true_task_num, self._false_task_num, self._true_success_count, self._false_success_count)
    
    # override
    def saveResults(self) -> str:

        """returns the directory path where results are saved"""

        dir_path = 'results/SimpleOpenAIBench/' + time.strftime("%y%m%d-%H.%M.%S")
        os.makedirs(dir_path, exist_ok=True)

        # save benchmark summary
        with open(f"{dir_path}/benchmark_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"Date and Time:\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            dataset_info = f"Dataset:\t{self._files_path['dataset']}" + \
                (f" (range: {self._files_path['dataset_range'][0]}-{self._files_path['dataset_range'][1]})\n" \
                    if "dataset_range" in self._files_path \
                    else "\n")
            f.write(dataset_info)

            f.write(f"Model Info:\n\tmodel_name: {self._model_info['model_name']}\n\tsampling_params: {self._model_info['sampling_params']}\n")
            f.write(f"Base Prompt:\t{self._files_path['simple_llm_bench']}\n")
            f.write(
                f"Accuracy:\t{(self._true_success_count+self._false_success_count)/max(self._true_task_num+self._false_task_num, 1):.2f} ({self._true_success_count+self._false_success_count}/{self._true_task_num+self._false_task_num})\n"
            )
            f.write(f"True Accuracy:\t{self._true_success_count/max(self._true_task_num, 1):.2f} ({self._true_success_count}/{self._true_task_num})\n")
            f.write(f"False Accuracy:\t{self._false_success_count/max(self._false_task_num, 1):.2f} ({self._false_success_count}/{self._false_task_num})\n")

        # save reasoning results
        reasoning_results = [
            {'ID': i, 'reasoning_results': self.__task_results[i]['reasoning_results']}
            for i in range(len(self.__task_results))
        ]
        with open(f"{dir_path}/reasoning_results.json", 'w', encoding='utf-8') as f:
            json.dump(reasoning_results, f, ensure_ascii=False, indent=2)

        # save search results
        search_results = [
            {'ID': i, 'search_results': self.__task_results[i]['search_results']}
            for i in range(len(self.__task_results))
        ]
        with open(f"{dir_path}/search_results.json", 'w', encoding='utf-8') as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)
        
        # save verification results
        verification_results = []
        for i, task_result in enumerate(self.__task_results):

            if isinstance(task_result['result'], bool):
            
                # 最終的な判定をチェック
                factuality = task_result['result']
                
                # 最終的な判定とタスクラベルを比較
                success = None
                if factuality == self._task_entities[i].label:
                    success = "succeeded"
                else:
                    success = "failed"

                # final_resultプロパティの値を作成
                final_result = f"{success} ({'True' if self._task_entities[i].label else 'False'} task)"

                verification_results.append({
                    'ID': i, 'final_result': final_result, 
                    'verification_result': {
                        'reason': task_result['reason'],
                        'factuality': factuality
                    }
                })

            else:
                verification_results.append({'ID': i, 'final_result': task_result['result'], 'verification_result': None})

        with open(f"{dir_path}/verification_results.json", 'w', encoding='utf-8') as f:
            json.dump(verification_results, f, ensure_ascii=False, indent=2)
        
        return dir_path

class AnthropicBenchControl(BenchmarkControl):

    __prompt_base: str = ""
    __task_results: List[dict] = [] # 結果の形式が違うため、TaskResultクラスは使えない

    def __init__(self, files_path: dict, model_info: dict):
        """
        **files_path**: dict with keys 'flowchart', 'decomposition', 'query', 'verification', 'verification_FC', 'dataset'  
        **model_info**: dict with keys 'model_name', 'sampling_params'  
        **sampling_params**: dict with keys for Anthropic API sampling parameters
        """
        super().__init__(files_path, model_info)
        self._anthropic_client = AsyncAnthropic()
    
    async def __requestAnthropic(self, prompt: str, max_attempt_num: int = 5) -> dict:
        
        """
        Returns dict with keys 'reason': str, 'result': bool, 'reasoning_results': List[str], 'search_results': dict.  
        Keys of search_results are queries, values are list of sources.
        """

        res = {}
        finished = False
        for i in range(max_attempt_num):

            print(f"==== {i+1}th attempt for Anthropic request ==== ") # debug

            response = await self._anthropic_client.messages.create(
                model = self._model_info["model_name"],
                messages = [
                    {"role": "user", "content": prompt}
                ],
                **self._model_info["sampling_params"]
            )
            
            contents = response.content
            reasoning_summaries: List[str] = []
            search_results: dict = {}
            query_table = {}
            response_text = ""
            for c in contents:
                
                # 推論の要約の取得
                if getattr(c, "type", None) == "thinking":
                    if getattr(c, "thinking", None) != None:
                        reasoning_summaries.append(c.thinking)
                
                # Web検索のクエリの取得
                if getattr(c, "type", None) == "server_tool_use":
                    if getattr(c, "input", None) != None and \
                        getattr(c, "id", None) != None and \
                        c.input.get("query", None) != None:
                        tool_use_id = c.id
                        query_table[tool_use_id] = c.input["query"]
                
                # Web検索の結果の取得
                if getattr(c, "type", None) == "web_search_tool_result":
                    if getattr(c, "tool_use_id", None) != None:
                        search_results[query_table[c.tool_use_id]] = []
                        if type(getattr(c, "content", None)) is list:
                            for page in c.content:
                                search_results[query_table[c.tool_use_id]].append({
                                    "title": getattr(page, "title", None),
                                    "url": getattr(page, "url", None)
                                })

                # テキストの取得
                if getattr(c, "type", None) == "text":
                    if getattr(c, "text", None) != None:
                        response_text += c.text

            res["search_results"] = search_results
            res["reasoning_results"] = reasoning_summaries

            matches = regex.findall(r"<json>(.*?)</json>", response_text, regex.DOTALL, overlapped=True)  # <json>と</json>の間を抜き出す（リーズニングの過程タグが出力され、マッチ範囲がネストした場合、再帰的に検索を行い最後のマッチ結果を取り出す）
            if len(matches) > 0:
                json_text = matches[-1].strip()
                try:
                    response_dict = json.loads(json_text) # jsonで読み込み、dictに変換する
                    if isinstance(response_dict, dict) and isinstance(response_dict.get("reason"), str) and isinstance(response_dict.get("result"), bool):
                        res["reason"] = response_dict["reason"]
                        res["result"] = response_dict["result"]
                        finished = True
                        break # 全てのフォーマットチェックをクリアしたので、試行ループを抜ける
                    else:
                        pass
                except json.JSONDecodeError:
                    pass
            else:
                pass

        if finished == False:
            raise RuntimeError("Anthropic request failed after maximum attempts.")

        return res

    def __checkCorrectNum(self):
        
        for i in range(len(self._task_entities)):

            label = self._task_entities[i].label

            # ラベルごとにタスク数を記録
            if label == True: self._true_task_num += 1
            if label == False: self._false_task_num += 1

            # 各タスクの最終的な真実性を取得
            factuality = None
            if isinstance(self.__task_results[i]['result'], bool):
                factuality = self.__task_results[i]['result']
            else:
                continue
            
            # ラベルごとに正当数記録
            if factuality == label:
                if label == True: self._true_success_count += 1
                if label == False: self._false_success_count += 1
    
    # override
    def loadBench(self):

        with open(self._files_path['simple_llm_bench'], encoding='utf-8') as f:
            self.__prompt_base = f.read()

        with open(self._files_path['dataset'], encoding='utf-8') as f:
            tasks = json.load(f)
            if 'dataset_range' in self._files_path: # コマンドライン引数でデータセットの範囲が指定されている場合
                tasks = tasks[self._files_path['dataset_range'][0]:self._files_path['dataset_range'][1]+1]
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
    
    # override
    async def runBench(self, on_task_done: Callable[[int, int], None]) -> Tuple[int]:
        
        """
        returns a tuple of (true_task_num, false_task_num, true_success_count, false_success_count)
        Results will be stored in self._task_results.
        on_task_done(finished_task_num: int, task_num: int) -> None
        """

        total = len(self._task_entities)
        self.__task_results = [None] * total # self._task_resultsでないことに注意

        open("bench_log.jsonl", "w", encoding='utf-8').close()  # ログファイルを初期化

        async def run_task(idx: int, task: TaskEntity) -> None:

            prompt = self.__prompt_base.replace(
                '<input-procedure>', task.procedure).replace(
                '<input-question>', task.user_question
            )
            
            try:
                task_result = await self.__requestAnthropic(prompt)
            except RuntimeError:
                raise
            
            self.__task_results[idx] = task_result
        
        tasks = []
        for idx, task in enumerate(self._task_entities):
            tasks.append(asyncio.create_task(run_task(idx, task)))

        exceptions = await asyncio.gather(*tasks, return_exceptions=True)
        for i, e in enumerate(exceptions):
            if isinstance(e, Exception):
                self.__task_results[i] = {
                    "reason": None, "result": str(e), "reasoning_results": None, "search_results": None
                }

        self.__checkCorrectNum()

        return (self._true_task_num, self._false_task_num, self._true_success_count, self._false_success_count)
    
    # override
    def saveResults(self) -> str:

        """returns the directory path where results are saved"""

        dir_path = 'results/SimpleAnthropicBench/' + time.strftime("%y%m%d-%H.%M.%S")
        os.makedirs(dir_path, exist_ok=True)

        # save benchmark summary
        with open(f"{dir_path}/benchmark_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"Date and Time:\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            dataset_info = f"Dataset:\t{self._files_path['dataset']}" + \
                (f" (range: {self._files_path['dataset_range'][0]}-{self._files_path['dataset_range'][1]})\n" \
                    if "dataset_range" in self._files_path \
                    else "\n")
            f.write(dataset_info)

            f.write(f"Model Info:\n\tmodel_name: {self._model_info['model_name']}\n\tsampling_params: {self._model_info['sampling_params']}\n")
            f.write(f"Base Prompt:\t{self._files_path['simple_llm_bench']}\n")
            f.write(
                f"Accuracy:\t{(self._true_success_count+self._false_success_count)/max(self._true_task_num+self._false_task_num, 1):.2f} ({self._true_success_count+self._false_success_count}/{self._true_task_num+self._false_task_num})\n"
            )
            f.write(f"True Accuracy:\t{self._true_success_count/max(self._true_task_num, 1):.2f} ({self._true_success_count}/{self._true_task_num})\n")
            f.write(f"False Accuracy:\t{self._false_success_count/max(self._false_task_num, 1):.2f} ({self._false_success_count}/{self._false_task_num})\n")
        
        # save reasoning results
        reasoning_results = [
            {'ID': i, 'reasoning_results': self.__task_results[i]['reasoning_results']}
            for i in range(len(self.__task_results))
        ]
        with open(f"{dir_path}/reasoning_results.json", 'w', encoding='utf-8') as f:
            json.dump(reasoning_results, f, ensure_ascii=False, indent=2)

        # save search results
        search_results = [
            {'ID': i, 'search_results': self.__task_results[i]['search_results']}
            for i in range(len(self.__task_results))
        ]
        with open(f"{dir_path}/search_results.json", 'w', encoding='utf-8') as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)
        
        # save verification results
        verification_results = []
        for i, task_result in enumerate(self.__task_results):

            if isinstance(task_result['result'], bool):
            
                # 最終的な判定をチェック
                factuality = task_result['result']
                
                # 最終的な判定とタスクラベルを比較
                success = None
                if factuality == self._task_entities[i].label:
                    success = "succeeded"
                else:
                    success = "failed"

                # final_resultプロパティの値を作成
                final_result = f"{success} ({'True' if self._task_entities[i].label else 'False'} task)"

                verification_results.append({
                    'ID': i, 'final_result': final_result, 
                    'verification_result': {
                        'reason': task_result['reason'],
                        'factuality': factuality
                    }
                })

            else:
                verification_results.append({'ID': i, 'final_result': task_result['result'], 'verification_result': None})

        with open(f"{dir_path}/verification_results.json", 'w', encoding='utf-8') as f:
            json.dump(verification_results, f, ensure_ascii=False, indent=2)
        
        return dir_path

# module test

if __name__ == "__main__":

    async def main():

        files_path = {
            'flowchart': 'prompts/prompt_flowchart0.0.0.txt',
            'decomposition': 'prompts/prompt_decompose5.7.3.txt',
            'common_query': 'prompts/prompt_query_common0.0.0.txt',
            'individual_query': 'prompts/prompt_query_individual0.0.0.txt',
            'verification': 'prompts/prompt_verify1.2.5.txt',
            'verificationFC': 'prompts/prompt_verifyFC0.0.0.txt',
            'dataset': 'dataset/test.json',
            'no_structure_decomposition': 'prompts/prompt_decompose_no_structure1.0.0.txt',
            'ontology_structure_decomposition': 'prompts/prompt_decompose_ontology0.0.0.txt',
            'simple_query': 'prompts/prompt_query_simple0.0.0.txt',
            'llm_bench': 'prompts/prompt_llm_bench0.0.0.txt'
        }

        benchmark_control = AnthropicBenchControl(files_path)
        
        benchmark_control.loadBench()

        def on_task_done(finished_task_num: int, task_num: int):

            # タスクの完了数を累積する静的メンバー
            # 関数が初めて呼び出されたときのみ0で初期化
            if not hasattr(on_task_done, "static_task_num"):
                on_task_done.static_task_num = 0

            on_task_done.static_task_num += 1

            print(f"====== {on_task_done.static_task_num}/{task_num} task(s) done ======")

            # ※finished_task_numは使わなくなった

        bench_result = await benchmark_control.runBench(on_task_done=on_task_done)
        benchmark_control.saveResults()

    asyncio.run(main())