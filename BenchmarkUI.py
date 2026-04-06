from BenchmarkControl import BenchmarkControl

class BenchmarkUI:

    __benchmark_control: BenchmarkControl

    def __init__(self, files_path: dict, model_info: dict, bench_type: str):
        
        match bench_type:
        
            case "openai_default":
                from OpenAIDefaultBenchControl import OpenAIDefaultBenchControl
                self.__benchmark_control = OpenAIDefaultBenchControl(files_path, model_info)

            case "vllm_default":
                from vLLMDefaultBenchControl import vLLMDefaultBenchControl
                self.__benchmark_control = vLLMDefaultBenchControl(files_path, model_info)

            case "no_decomposition":
                from NoDecompositionBenchControl import NoDecompositionBenchControl
                self.__benchmark_control = NoDecompositionBenchControl(files_path, model_info)

            case "no_structure":
                from NoStructureBenchControl import NoStructureBenchControl
                self.__benchmark_control = NoStructureBenchControl(files_path, model_info)
            case "no_individual_query":
                from NoIndividualQueryBenchControl import NoIndividualQueryBenchControl
                self.__benchmark_control = NoIndividualQueryBenchControl(files_path, model_info)

            case "ontology_structure":
                from OntologyStructureBenchControl import OntologyStructureBenchControl
                self.__benchmark_control = OntologyStructureBenchControl(files_path, model_info)

            case "simple_query":
                from SimpleCommonQueryBenchControl import SimpleCommonQueryBenchControl
                self.__benchmark_control = SimpleCommonQueryBenchControl(files_path, model_info)
            
            case "simple_openai_bench":
                from LLMBenchControl import OpenAIBenchControl
                self.__benchmark_control = OpenAIBenchControl(files_path, model_info)

            case "simple_anthropic_bench":
                from LLMBenchControl import AnthropicBenchControl
                self.__benchmark_control = AnthropicBenchControl(files_path, model_info)

            case _:
                raise ValueError(f"Unknown benchmark type: {bench_type}")
            
    async def startBench(self):

        """
        asynchronous method
        """

        # ベンチマークのロード
        print("Start loading benchmark...", end=' ')
        self.__benchmark_control.loadBench()
        print("Benchmark loaded.")

        def on_task_done(finished_task_num: int, task_num: int):

            # タスクの完了数を累積する静的メンバー
            # 関数が初めて呼び出されたときのみ0で初期化
            if not hasattr(on_task_done, "static_task_num"):
                on_task_done.static_task_num = 0

            on_task_done.static_task_num += 1

            print(f"====== {on_task_done.static_task_num}/{task_num} task(s) done ======")

            # ※finished_task_numは使わなくなった

        # ベンチマークの実行
        print("Start running benchmark...")
        bench_result = await self.__benchmark_control.runBench(on_task_done=on_task_done)
        print("Benchmark finished.")


        # 結果の概要の表示
        true_task_num, false_task_num, true_success_count, false_success_count = bench_result
        total_accuracy = (true_success_count + false_success_count) / max((true_task_num + false_task_num), 1)
        true_accuracy = true_success_count / max(true_task_num, 1)
        false_accuracy = false_success_count / max(false_task_num, 1)
        print("\n==============Results Summary==============")
        print(f"Total accuracy: {total_accuracy:.2f} ({true_success_count + false_success_count}/{true_task_num + false_task_num})")
        print(f"Accuracy for true-labeled tasks: {true_accuracy:.2f} ({true_success_count}/{true_task_num})")
        print(f"Accuracy for false-labeled tasks: {false_accuracy:.2f} ({false_success_count}/{false_task_num})")
        print("===========================================\n")

        # 結果の保存
        print("Start saving results...", end=' ')
        dir_path = self.__benchmark_control.saveResults()
        print(f"Results saved to {dir_path}")