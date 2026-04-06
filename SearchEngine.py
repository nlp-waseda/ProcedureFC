from abc import ABC, abstractmethod
import aiohttp
import asyncio
from typing import List
from bs4 import BeautifulSoup
import re
from protego import Protego

class SearchEngine(ABC):

    def __init__(self, client):

        """
        region: us-en, jp-jp, etc.
        """

        self._client = client
        self.__region = 'us-en'
        self.__AGENT = 'my-app/0.0.1'

    async def __is_allowed_by_robots(self, href: str, user_agent: str='*') -> bool:
        
        # 正規表現でURLからrobots.txtのURLを作成
        robots_url = re.findall(r'^(https?://[^/]+)', href)[0] + '/robots.txt'

        # robots.txtの内容を取得（rootperserのreadだと取得できない時がある）
        response = None
        try:
            async with aiohttp.ClientSession(headers={'user-agent': self.__AGENT}) as session:
                async with session.get(robots_url) as response:
                    robots_txt = await response.text()
        except Exception:
            # robots.txtが取得できなかった場合は許可（True）を返す
            return True

        # robots.txtの内容をパース
        rp = Protego.parse(robots_txt)

        return rp.can_fetch(href, user_agent)

    @abstractmethod
    async def _requestSE(self, query: str, page_num: int):
        """
        return list of dicts with keys: title, href
        """

    
    async def search(self, query: str, page_num: int = 2) -> List[dict]:

        """
        return list of dicts with keys: title, href, text
        """
        
        results = await self._requestSE(query, page_num)

        # resultsからタイトルとリンクをリストに格納
        pages = [
            {
                'title': result['title'],
                'href': result['href']
            } for result in results
        ]        

        re = []
        page_count = 0
        for p in pages:

            if page_count >= page_num:
                break

            # robots.txtを確認して、クロールが許可されているかを確認する
            allowed = await self.__is_allowed_by_robots(p['href'], user_agent=self.__AGENT)
            if not allowed:
                # print(f"Skipping {p['href']} due to robots.txt restrictions.")
                continue

            # HTMLを取得
            try:
                async with aiohttp.ClientSession(headers={'user-agent': self.__AGENT}) as session:
                    async with session.get(p['href']) as response:
                        html = await response.text()
            except Exception:
                # print(f"Skipping {p['href']} due to access error.")
                continue

            # BeautifulSoupでパース
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style"]): # scriptやstyleを含む要素を削除する
                script.decompose()
            text=soup.get_text() # テキストのみを取得=タグは全部取る
            lines= [line.strip() for line in text.splitlines()] # textを改行ごとにリストに入れて、リスト内の要素の前後の空白を削除
            text="\n".join(line for line in lines if line) # リストの空白要素以外をすべて文字列に戻す

            # 情報量の少ないページ（JSが必要なページなど）は除外
            if len(text) < 300:
                # print(f"Skipping {p['href']} due to insufficient content.")
                continue

            re.append({'title': p['title'], 'href': p['href'],'text': text})

            page_count += 1

        return re

# main
if __name__ == "__main__":

    async def main():

        search_engine = SearchEngine('jp-jp')
        
        query = "原神 良茶満月 入手方法"
        results = await search_engine.search(query, 4)
        for (k, v) in enumerate(results, 1):
            print(f"{k}. {v['title']} <{v['href']}>\n{v['text']}\n")

    asyncio.run(main())