import re

from bs4 import BeautifulSoup


class Parser:
    
    @classmethod
    async def parse(cls, html_text: str) -> str:
        soup = BeautifulSoup(html_text, 'html.parser')
        
        return re.sub(r'[\n\r\t]', ' ', soup.get_text()).strip()