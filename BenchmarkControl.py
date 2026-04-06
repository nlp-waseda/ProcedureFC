from typing import List, Callable, Tuple
import json
import asyncio
import time
import os
from abc import ABC, abstractmethod

from DataType import *
from LLMModel import LLMModel
from FlowchartAgent import FlowchartAgent
from DecompositionAgent import DecompositionAgent
from QueryGenerator import QueryGenerator
from VerificationAgent import VerificationAgent
from SearchEngine import SearchEngine
from GoogleSE import GoogleSE

class BenchmarkControl(ABC):

    _files_path: dict = None
    _model_info: dict = None

    _task_entities: List[TaskEntity] = []
    _task_results: List[TaskResult] = []
    
    _true_task_num = 0
    _false_task_num = 0
    _true_success_count = 0
    _false_success_count = 0

    _llm_model: LLMModel = None

    _flowchart_agent = None
    _decomposition_agent = None
    _query_generator = None
    _verification_agent = None
    
    _search_engine: SearchEngine = None

    def __init__(self, files_path: dict, model_info: dict):
        """
        **files_path**: dict with keys 'flowchart', 'decomposition', 'query', 'verification', 'verification_FC', 'dataset'  
        **model_info**: dict with keys 'model_name', 'sampling_params'
        """
        self._files_path = files_path
        self._model_info = model_info
        self._search_engine = GoogleSE()

    @abstractmethod
    def loadBench(self):
        pass

    @abstractmethod
    async def runBench(self, on_task_done: Callable[[int, int], None]) -> Tuple[int]:
        """
        returns a tuple of (true_task_num, false_task_num, true_success_count, false_success_count)
        """
        pass

    @abstractmethod
    def saveResults(self) -> str:
        """returns the directory path where results are saved"""
        pass

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

        benchmark_control = BenchmarkControl(files_path)

        print("Start loading default benchmark...")
        benchmark_control.loadDefaultBench()
        print("Default benchmark loaded.")

        def on_task_done(finished_task_num: int, task_num: int):

            # タスクの完了数を累積する静的メンバー
            # 関数が初めて呼び出されたときのみ0で初期化
            if not hasattr(on_task_done, "static_task_num"):
                on_task_done.static_task_num = 0

            on_task_done.static_task_num += 1

            print(f"====== {on_task_done.static_task_num}/{task_num} task(s) done ======")

            # ※finished_task_numは使わなくなった

        print("Start running default benchmark...")
        await benchmark_control.runDefaultBench(on_task_done=on_task_done)
        print("Default benchmark finished.")

        print("Start saving results...")
        dir_path = benchmark_control.saveResults()
        print(f"Results saved to {dir_path}")

    asyncio.run(main())