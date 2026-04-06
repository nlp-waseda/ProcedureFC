import os
import sys
from typing import List
import ast

from FactcheckControl import FactcheckControl

class FactcheckUI:

    __factcheck_control: FactcheckControl = None

    def __checkEnvs(self) -> List[str]:
        required_envs = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CSE_ID"]
        missing = [env for env in required_envs if not os.environ.get(env)]
        return missing

    def __init__(self, files_path):

        # 不足している環境変数がある場合エラー終了
        missing_envs = self.__checkEnvs()
        if missing_envs:
            print("Missing environment variables:\n", file=sys.stderr)
            for env in missing_envs:
                print(f" - {env}", file=sys.stderr)
            sys.exit(1)

        self.__factcheck_control = FactcheckControl(files_path)

    def __inputUserQuestion(self) -> str:
        user_question = input("\nEnter user's question: ")
        if user_question.strip() == "":
            print("User question cannot be empty.")
            return self.__inputUserQuestion()
        else:
            converted = ast.literal_eval(f'"{user_question}"')
            return converted

    def __inputAIResponse(self) -> str:
        procedure = input("\nEnter AI's response (procedure): ")
        if procedure.strip() == "":
            print("AI response cannot be empty.")
            return self.__inputAIResponse()
        else:
            converted = ast.literal_eval(f'"{procedure}"')
            return converted

    def __showResults(self, false_statements: List[dict]):
        print("\n===Factcheck Result===")
        if not false_statements:
            print("    True")
        else:
            print("    False")
        print("======================")

        if false_statements:
            print("\n<Incorrect Statements>")
            for statement in false_statements:
                print("---------------------------------------------------------------")
                print(f"\"{statement['original_sentence']}\"")
                print(f"\n=> {statement['reason']}\n")

    async def startFactcheck(self):
        

        user_question = self.__inputUserQuestion()
        procedure = self.__inputAIResponse()

        self.__factcheck_control.loadFactcheck()

        print("\nStart Factcheck\n")

        def indicator(finished: int):

            tasks = ["Decomposition", "SearchQuery Generation", "Web Search", "Verification"]

            # Initialize status as a member variable
            if not hasattr(indicator, "indicator_status"):
                indicator.indicator_status = ["[ ]" for _ in range(4)]
                print("\n"*len(tasks), end="")

            indicator.indicator_status[finished] = "[x]"
            print("\033[F" * len(tasks), end="")  # Move cursor up to overwrite the checklist
            for i, task in enumerate(tasks):
                print(f"{indicator.indicator_status[i]} {task}")

        false_statements = await self.__factcheck_control.factcheck(user_question, procedure, indicator)

        print("\nFactcheck complete\n")

        self.__showResults(false_statements)