import requests
from bs4 import BeautifulSoup

res = requests.get('https://cs.nyu.edu/~jversoza/dma/final2022/test.html')
dom = BeautifulSoup(res.text, "html.parser")
rows = dom.find_all('p', attrs={'class': 'info'})
for row in rows:
    line = row.find_all('li')
    if (float(line[1].text) > 4):
        print('{0} BY {1}: {2}'.format(row.find('h2').text, line[0].text, line[1].text))
