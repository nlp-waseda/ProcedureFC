class TaskEntity:

    def __init__(self, ID: int, user_question: str, procedure: str, label: bool, supplementary_info: str):
        self.__ID = ID
        self.__user_question = user_question
        self.__procedure = procedure
        self.__label = label
        self.__supplementary_info = supplementary_info
    
    # @property getters
    @property
    def ID(self) -> int:
        return self.__ID
    @property
    def user_question(self) -> str:
        return self.__user_question
    @property
    def procedure(self) -> str:
        return self.__procedure
    @property
    def label(self) -> bool:
        return self.__label
    @property
    def supplementary_info(self) -> str:
        return self.__supplementary_info
    
from typing import List

class TaskResult:

    def __init__(self, ID: int, flowchart: str, statements: dict, queries: dict, evidences: List[dict], verification_results: List[dict]):
        """
        dictionary with keys:
        - statements: number, sentence
        - queries: c_queries, i_queries
        - evidence: title, href, text
        - verification_result: reason, result
        """
        self.__ID = ID
        self.__flowchart = flowchart
        self.__statements = statements
        self.__queries = queries
        self.__evidences = evidences
        self.__verification_results = verification_results
    
    # @property Getters
    @property
    def ID(self) -> int:
        return self.__ID
    @property
    def flowchart(self) -> str:
        return self.__flowchart
    @property
    def statements(self) -> dict:
        """keys: number, sentence"""
        return self.__statements
    @property
    def queries(self) -> dict:
        """keys: c_queries, i_queries"""
        return self.__queries
    @property
    def evidences(self) -> List[dict]:
        """keys: title, href, text"""
        return self.__evidences
    @property
    def verification_results(self) -> List[dict]:
        """keys: reason, result"""
        return self.__verification_results