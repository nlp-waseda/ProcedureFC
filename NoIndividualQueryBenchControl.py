import asyncio
from typing import List, Callable, Tuple
import json
import time
import os

from OpenAIAPI import OpenAIAPI
from FlowchartAgent import FlowchartAgent
from DecompositionAgent import DecompositionAgent
from QueryGenerator import QueryGenerator
from VerificationAgent import VerificationAgent
from BenchmarkControl import BenchmarkControl
from DataType import *

class NoIndividualQueryBenchControl(BenchmarkControl):

    def __init__(self, files_path: dict, model_info: dict):
        """
        **files_path**: dict with keys 'flowchart', 'decomposition', 'query', 'verification', 'verification_FC', 'dataset'  
        **model_info**: dict with keys 'model_name', 'sampling_params'  
        **sampling_params**: dict with keys for OpenAI API sampling parameters (e.g., temperature)
        """
        super().__init__(files_path, model_info)

    # override
    def loadBench(self):

        self._llm_model = OpenAIAPI(self._model_info["model_name"], self._model_info["sampling_params"])

        with open(self._files_path['flowchart'], encoding='utf-8') as f:
            prompt_base = f.read()
            self._flowchart_agent = FlowchartAgent(self._llm_model, prompt_base, 5)

        with open(self._files_path['decomposition'], encoding='utf-8') as f:
            prompt_base = f.read()
            self._decomposition_agent = DecompositionAgent(self._llm_model, prompt_base, 5)
        prompt_base = []
        with open(self._files_path['common_query'], encoding='utf-8') as f:
            prompt_base.append(f.read())
        with open(self._files_path['individual_query'], encoding='utf-8') as f:
            prompt_base.append(f.read())
        self._query_generator = QueryGenerator(self._llm_model, prompt_base, 5)

        prompt_base = []
        with open(self._files_path['verification'], encoding='utf-8') as f:
            prompt_base.append(f.read())
        with open(self._files_path['verificationFC'], encoding='utf-8') as f:
            prompt_base.append(f.read())
        self._verification_agent = VerificationAgent(self._llm_model, prompt_base, 5)

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

    def __checkCorrectNum(self):
        
        for i in range(len(self._task_entities)):

            label = self._task_entities[i].label

            # ラベルごとにタスク数を記録
            if label == True: self._true_task_num += 1
            if label == False: self._false_task_num += 1

            # 各タスクの最終的な真実性を取得
            factuality = True
            step_results = self._task_results[i].verification_results
            if isinstance(step_results, list):
                for r in step_results:
                    if r['result'] == False:
                        factuality = False
                        break
            else:
                continue # タスク実行中に例外が発生した場合
            
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

        open("bench_log.jsonl", "w", encoding='utf-8').close()  # ログファイルを初期化

        async def run_task(idx: int, task: TaskEntity) -> None:

            # フローチャート生成タスク
            fc_task = asyncio.create_task(self._flowchart_agent.createFlowchart(task.procedure))
            
            # 共通クエリ生成タスク
            c_query_task = asyncio.create_task(self._query_generator.generateCommonQueries(task.user_question, task.procedure))

            # 分解タスク
            decompose_task = asyncio.create_task(self._decomposition_agent.decompose(task.procedure))

            # 個別クエリ生成タスクなし

            # Web検索タスク
            try:
                c_queries = await c_query_task
            except RuntimeError:
                raise
            queries = set(c_queries) # クエリの重複を排除
            queries = list(queries)
            pages = {}
            async def r(q: str):
                pages[q] = await self._search_engine.search(q)
            search_tasks = []
            for q in queries:
                search_tasks.append(asyncio.create_task(r(q)))

            # 検証タスク
            try:
                flowchart, decompose_result = await asyncio.gather(fc_task, decompose_task)
            except RuntimeError:
                raise
            verify_tasks = []
            await asyncio.gather(*search_tasks)
            ## フローチャートの検証
            fc_evidence = ''
            for q in c_queries:
                for p in pages[q]:
                    fc_evidence += f"<title>\n{p['title']}\n<body>\n{p['text']}\n\n"
            verify_tasks.append(asyncio.create_task(self._verification_agent.verifyFC(fc_evidence, flowchart, max_evidence_length=1680000))) # max_evidence_length: (gpt-5.1)350,000*4.8=1,680,000
            ## ステートメントの検証
            for i, statement in enumerate(decompose_result):
                # 証拠の結合
                evidence = ''
                for q in c_queries:
                    for p in pages[q]:
                        evidence += f"<title>\n{p['title']}\n<body>\n{p['text']}\n\n"
                verify_tasks.append(asyncio.create_task(self._verification_agent.verify(evidence, statement['sentence'], max_evidence_length=1680000))) # max_evidence_length: (gpt-5.1)350,000*4.8=1,680,000

            try:
                verify_results = await asyncio.gather(*verify_tasks)
            except RuntimeError:
                for vt in verify_tasks:
                    vt.cancel() # verifyタスクは数が多いので、1つでも例外が発生したら、明示的に他のタスクをキャンセルする
                await asyncio.gather(*verify_tasks, return_exceptions=True)
                raise

            queries_dict = {'c_queries': c_queries}
            self._task_results[idx] = TaskResult(task.ID, flowchart, decompose_result, queries_dict, pages, verify_results)

            # エラー発生の可能性を考え、リアルタイムでログを生成
            with open("bench_log.jsonl", "a", encoding='utf-8') as f:
                json_line = json.dumps({
                    'ID': task.ID,
                    'flowchart': self._task_results[idx].flowchart,
                    'decomposition': self._task_results[idx].statements,
                    'queries': self._task_results[idx].queries,
                    'evidences': self._task_results[idx].evidences,
                    'verification_results': self._task_results[idx].verification_results
                }, ensure_ascii=False)
                f.write(json_line + "\n")
            
            on_task_done(idx, total)

        tasks = []
        for idx, task in enumerate(self._task_entities):
            tasks.append(asyncio.create_task(run_task(idx, task)))
        
        exceptions = await asyncio.gather(*tasks, return_exceptions=True)
        for i, e in enumerate(exceptions):
            if isinstance(e, Exception):
                self._task_results[i] = TaskResult(i, None, None, None, None, repr(e)) # エラーが起きたタスクのtask_resultはNoneとError Objectで埋める

        self.__checkCorrectNum()

        return (self._true_task_num, self._false_task_num, self._true_success_count, self._false_success_count)

    def saveResults(self) -> str:

        """returns the directory path where results are saved"""

        dir_path = 'results/NoIndividualQueryBench/' + time.strftime("%y%m%d-%H.%M.%S")
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
            f.write(f"Flowchart Prompt:\t{self._files_path['flowchart']}\n")
            f.write(f"Decomposition Prompt:\t{self._files_path['decomposition']}\n")
            f.write(f"Query Prompt:\t(common) {self._files_path['common_query']} (individual) None\n")
            f.write(f"Verification Prompt:\t(statement) {self._files_path['verification']} (flowchart) {self._files_path['verificationFC']}\n")
            f.write(
                f"Accuracy:\t{(self._true_success_count+self._false_success_count)/max(self._true_task_num+self._false_task_num, 1):.2f} ({self._true_success_count+self._false_success_count}/{self._true_task_num+self._false_task_num})\n"
            )
            f.write(f"True Accuracy:\t{self._true_success_count/max(self._true_task_num, 1):.2f} ({self._true_success_count}/{self._true_task_num})\n")
            f.write(f"False Accuracy:\t{self._false_success_count/max(self._false_task_num, 1):.2f} ({self._false_success_count}/{self._false_task_num})\n")

        # save flowchart results
        flowchart_results = [
            {'ID': result.ID, 'result': result.flowchart}
            for result in self._task_results
        ]
        with open(f"{dir_path}/flowchart_results.json", 'w', encoding='utf-8') as f:
            json.dump(flowchart_results, f, ensure_ascii=False, indent=2)

        # save decomposition results
        decomposition_results = [
            {'ID': result.ID, 'result': result.statements}
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
        verification_results = []
        for i, task_result in enumerate(self._task_results):

            if isinstance(task_result.verification_results, list):

                # verifyの最終的な判定をチェック
                factuality = True
                for v in task_result.verification_results:
                    if v['result'] == False:
                        factuality = False
                        break
                
                # 最終的な判定とタスクラベルを比較
                success = None
                if factuality == self._task_entities[i].label:
                    success = "succeeded"
                else:
                    success = "failed"

                # final_resultプロパティの値を作成
                final_result = f"{success} ({'True' if self._task_entities[i].label else 'False'} task)"

                verification_results.append({'ID': task_result.ID, 'final_result': final_result, 'verification_results': task_result.verification_results})

            else:
                verification_results.append({'ID': task_result.ID, 'final_result': task_result.verification_results, 'verification_results': None})

        with open(f"{dir_path}/verification_results.json", 'w', encoding='utf-8') as f:
            json.dump(verification_results, f, ensure_ascii=False, indent=2)

        return dir_path

# module test
if __name__ == "__main__":

    async def main():

        print("Hello")
    
        files_path = {
            'decomposition': 'prompts/prompt_decompose5.2.txt',
            'query': 'prompts/prompt_query0.0.txt',
            'verification': 'prompts/prompt_verify1.1.2.txt',
            'dataset': 'dataset/dataset0.1.0.json'
        }

        benchmark_control = NoIndividualQueryBenchControl(files_path)

        print("Start loading no structure benchmark...")
        benchmark_control.loadBench()
        print("No structure benchmark loaded.")

        def on_task_done(finished_task_num: int, task_num: int):

            # タスクの完了数を累積する静的メンバー
            # 関数が初めて呼び出されたときのみ0で初期化
            if not hasattr(on_task_done, "static_task_num"):
                on_task_done.static_task_num = 0

            on_task_done.static_task_num += 1

            print(f"====== {on_task_done.static_task_num}/{task_num} task(s) done ======")

            # ※finished_task_numは使わなくなった

        print("Start running default benchmark...")
        await benchmark_control.runBench(on_task_done=on_task_done)
        print("Default benchmark finished.")

        print("Start saving results...")
        dir_path = benchmark_control.saveResults()
        print(f"Results saved to {dir_path}")

    asyncio.run(main())