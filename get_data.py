import requests
from bs4 import BeautifulSoup
import time
import random
import jieba
import numpy as np
from snownlp import SnowNLP
import matplotlib.pyplot as plt
from wordcloud import WordCloud

###Get Data
header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36"}
f_cookies = open('cookie.txt', 'r')
cookies = {}
for line in f_cookies.read().split(';'):
    name, value = line.strip().split('=', 1)
    cookies[name] = value
print(cookies)

comments = []
for page in range(24):
    url = f'https://movie.douban.com/subject/27010768/comments?start={20 * page}&limit=20&sort=new_score&status=P'
    res = requests.get(url, headers = header, cookies = cookies)
    soup = BeautifulSoup(res.text, 'lxml')
    comments += [element.p.span.string for element in soup.findAll(class_ = 'comment-item')]

comments= [x for x in comments if x is not None]

with open('寄生虫.txt', 'w', encoding='utf-8') as f:
    f.write('\n———————————————\n'.join(comments))


### Emotion Analysis
f = open('寄生虫.txt', 'r', encoding='UTF-8')
list = f.readlines()
sentimentslist = []
for i in list:
    s = SnowNLP(i)
    # print s.sentiments
    sentimentslist.append(s.sentiments)
plt.hist(sentimentslist, bins=np.arange(0, 1, 0.01), facecolor='g')
plt.xlabel('Sentiments Probability')
plt.ylabel('Quantity')
plt.title('Analysis of Sentiments')
plt.show()

###  Bulid wordcloud
with open('stopwords.txt', 'r') as f:
    stopwords = {}.fromkeys(f.read().split('\n'))

text = '\n\n'.join(comments)
result = [word for word in jieba.cut(text) if len(word) > 1 and word not in stopwords]
result = ",".join(result)

wc = WordCloud(
    background_color="white",  # 背景颜色
    max_words=100,  # 显示最大词数
    font_path='C:\Windows\Fonts\SIMYOU.TTF',  # 使用字体
    min_font_size=10,
    max_font_size=60,
    width=800  # 图幅宽度
)
wc.generate(result)
wc.to_file("寄生虫.png")
