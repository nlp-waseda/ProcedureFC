import asyncio
import sys

from BenchmarkUI import BenchmarkUI
from FactcheckUI import FactcheckUI

async def main():

    print("Hello.")

    files_path = {

        # ==== You can set your own dataset here !! ====
        'dataset': 'dataset/dataset0.5.3.json',

        'flowchart': 'prompts/prompt_flowchart0.0.0.txt',
        'decomposition': 'prompts/prompt_decompose5.7.3.txt',
        'common_query': 'prompts/prompt_query_common0.0.1.txt',
        'individual_query': 'prompts/prompt_query_individual0.0.0.txt',
        'verification': 'prompts/prompt_verify1.2.6.txt',
        'verificationFC': 'prompts/prompt_verifyFC0.0.1.txt',
        'no_structure_decomposition': 'prompts/prompt_decompose_no_structure1.0.0.txt',
        'ontology_structure_decomposition': 'prompts/prompt_decompose_ontology0.0.1.txt',
        'simple_query': 'prompts/prompt_query_simple0.0.0.txt',
        'simple_llm_bench': 'prompts/prompt_simple_llm_bench0.0.0.txt'
    }
    
    model_infos = {

        # ==== Models (& Sampling Parameters) provided by OpenAI API can be choosen ====
        "openai_default_bench": {
            "model_name": "gpt-5.1",
            "sampling_params": {
                "temperature": 0.5
            }
        },

        # ==== Models (& Sampling Parameters) provided by HuggingFace (and compatible with vLLM) can be choosen ====
        "vllm_default_bench": {
            "model_name": "openai/gpt-oss-20b", # 20b or 120b
            "sampling_params": {
                "temperature": 0.5,
                "top_p": 1,
                "top_k": -1,
                "max_tokens": 20000
            }
        },
        
        #"vllm_default_bench": {
        #    "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        #    "sampling_params": {
        #        "temperature": 0.5,
        #        "top_p": 0.8,
        #        "top_k": 20,
        #        "max_tokens": 6000,
        #        "min_p": 0
        #    }
        #},
        
        #"simple_openai_bench": {
        #    "model_name" : "gpt-5.1",
        #    "sampling_params" : {
        #        "tools" : [{"type": "web_search"}],
        #        "reasoning" : {"effort": "high", "summary": "detailed"},
        #        "include" : ["web_search_call.action.sources"]
        #    }
        #},
        "simple_openai_bench": {
            "model_name" : "gpt-5.1",
            "sampling_params" : {
                "tools" : [{"type": "web_search"}],
                "include" : ["web_search_call.action.sources"],
                "temperature": 0.5
            }
        },
        
        #"simple_anthropic_bench": {
        #    "model_name" : "claude-sonnet-4-5",
        #    "sampling_params" : {
        #        "max_tokens" : 20000,
        #        "tools" : [{"name": "web_search", "type": "web_search_20250305"}],
        #        "thinking" : {"type": "enabled", "budget_tokens": 19000}
        #    }
        #}
        "simple_anthropic_bench": {
            "model_name" : "claude-sonnet-4-5",
            "sampling_params" : {
                "max_tokens" : 20000,
                "tools" : [{"name": "web_search", "type": "web_search_20250305"}],
                "temperature": 0.5
            }
        }
    }

    if len(sys.argv) == 1:
        factcheck_ui = FactcheckUI(files_path)
        await factcheck_ui.startFactcheck()

    if len(sys.argv) > 1 and sys.argv[1] == "--openai-default-bench":
        if len(sys.argv) == 4:
            files_path["dataset_range"] = [int(sys.argv[2]), int(sys.argv[3])]
        benchmark_ui = BenchmarkUI(files_path, model_infos["openai_default_bench"], "openai_default")
        await benchmark_ui.startBench()

    if len(sys.argv) > 1 and sys.argv[1] == "--vllm-default-bench":
        if len(sys.argv) == 4:
            files_path["dataset_range"] = [int(sys.argv[2]), int(sys.argv[3])]
        benchmark_ui = BenchmarkUI(files_path, model_infos["vllm_default_bench"], "vllm_default")
        await benchmark_ui.startBench()

    if len(sys.argv) > 1 and sys.argv[1] == "--no-decomposition-bench":
        if len(sys.argv) == 4:
            files_path["dataset_range"] = [int(sys.argv[2]), int(sys.argv[3])]
        benchmark_ui = BenchmarkUI(files_path, model_infos["openai_default_bench"], "no_decomposition")
        await benchmark_ui.startBench()

    if len(sys.argv) > 1 and sys.argv[1] == "--no-structure-bench":
        if len(sys.argv) == 4:
            files_path["dataset_range"] = [int(sys.argv[2]), int(sys.argv[3])]
        benchmark_ui = BenchmarkUI(files_path, model_infos["openai_default_bench"], "no_structure")
        await benchmark_ui.startBench()

    if len(sys.argv) > 1 and sys.argv[1] == "--no-individual-query-bench":
        if len(sys.argv) == 4:
            files_path["dataset_range"] = [int(sys.argv[2]), int(sys.argv[3])]
        benchmark_ui = BenchmarkUI(files_path, model_infos["openai_default_bench"], "no_individual_query")
        await benchmark_ui.startBench()

    if len(sys.argv) > 1 and sys.argv[1] == "--ontology-structure-bench":
        if len(sys.argv) == 4:
            files_path["dataset_range"] = [int(sys.argv[2]), int(sys.argv[3])]
        benchmark_ui = BenchmarkUI(files_path, model_infos["openai_default_bench"], "ontology_structure")
        await benchmark_ui.startBench()

    if len(sys.argv) > 1 and sys.argv[1] == "--simple-query-bench":
        if len(sys.argv) == 4:
            files_path["dataset_range"] = [int(sys.argv[2]), int(sys.argv[3])]
        benchmark_ui = BenchmarkUI(files_path, model_infos["simple_openai_bench"], "simple_query")
        await benchmark_ui.startBench()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simple-openai-bench":
        if len(sys.argv) == 4:
            files_path["dataset_range"] = [int(sys.argv[2]), int(sys.argv[3])]
        benchmark_ui = BenchmarkUI(files_path, model_infos["simple_openai_bench"], "simple_openai_bench")
        await benchmark_ui.startBench()

    if len(sys.argv) > 1 and sys.argv[1] == "--simple-anthropic-bench":
        if len(sys.argv) == 4:
            files_path["dataset_range"] = [int(sys.argv[2]), int(sys.argv[3])]
        benchmark_ui = BenchmarkUI(files_path, model_infos["simple_anthropic_bench"], "simple_anthropic_bench")
        await benchmark_ui.startBench()

if __name__ == "__main__":

    asyncio.run(main())
