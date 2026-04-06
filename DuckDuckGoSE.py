from SearchEngine import SearchEngine

class DuckDuckGoSE(SearchEngine):
    def __init__(self):
        super().__init__("DuckDuckGo")

# results = self.__d.text(query, max_results=page_num+10, region=self.__region, safesearch='moderate')