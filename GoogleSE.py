import os
import asyncio
from googleapiclient.discovery import build
from SearchEngine import SearchEngine

class GoogleSE(SearchEngine):

    __api_key: str = None
    __cse_id: str = None
    __QUERIES_PER_MINUTE: int = 100 # Change this according to your API plan
    __request_ticket: int = 0 # APIにリクエストを投げるためには、1クエリにつき1枚このチケットが必要。トークンバケツのようなイメージ
    
    def __init__(self):

        self.__api_key = os.environ["GOOGLE_API_KEY"]
        self.__cse_id = os.environ["GOOGLE_CSE_ID"]
        self.__request_ticket = self.__QUERIES_PER_MINUTE

        client = build("customsearch", "v1", developerKey=self.__api_key).cse()
        
        super().__init__(client)

        asyncio.create_task(self.__refill_tickets())

    async def __refill_tickets(self):

        while True:
            await asyncio.sleep(70) # 念のため1分+10秒待つ
            self.__request_ticket = self.__QUERIES_PER_MINUTE

    # override
    async def _requestSE(self, query: str, page_num: int):

        while True:

            query_words = len(query.split(" "))

            if self.__request_ticket >= query_words:

                self.__request_ticket -= query_words
                
                response = self._client.list(
                    q=query,
                    cx=self.__cse_id,
                    num=10 # Google Custom Search APIは10ページがMAX
                ).execute()
                items = response.get('items', [])
                # 'link'キーを'href'キーに置き換えたdictリストを作成
                result = []
                for item in items:
                    d = dict(item)  # 元のdictをコピー
                    if 'link' in d:
                        d['href'] = d.pop('link')
                    result.append(d)
                
                return result
            
            else:
                await asyncio.sleep(5)

# module test
"""
if __name__ == "__main__":
    
    async def main():
        google_se = GoogleSE()
        queries = [f"Query-{i}" for i in range(110)]
        tasks = [asyncio.create_task(google_se.requestSE(query,2)) for query in queries]
        results = await asyncio.gather(*tasks)
        pass
    
    asyncio.run(main())
"""