import requests
import re
from bs4 import  BeautifulSoup


def getHTMLText(url):
    try:
        r = requests.get(url,timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""


def parsePage(ilt, html,good,count):
    try:
        soup=BeautifulSoup(html,"html.parser")
        demo=soup.find_all(class_='pub')
        til=re.findall(r'title=\".*?\"',html)
        for tt in range(len(demo)) :
            content=demo[tt].string.split('/')
            book=eval(til[tt].split('=')[1])
            auth=content[0]
            pub=content[1]
            date=content[2]
            price=content[3]
            ilt.append([book,auth,pub,date,price])
        img = soup.find_all(class_='nbg')
        for i in range(len(img)):
            try:
                src = img[i]('img')[0].attrs['src']
                pic=requests.get(src,timeout=10)
                dir = 'images/' + good + '_' + str(count) + '.jpg'
                fp = open(dir, 'wb')
                fp.write(pic.content)
                fp.close()
                count = count + 1
            except requests.exceptions.ConnectionError:
                print("第" + count + "张图片下载失败")
                continue
    except:
        print("")

def printGoodsList(ilt):
    tplt = "{:3}\t{:8}\t{:6}\t{:10}\t{:6}\t{:6}"
    print(tplt.format("序号","书名","作者", "出版商", "出版日期","价格"))
    count=0
    for g in ilt:
        count = count + 1
        print(tplt.format(count,g[0],g[1],g[2],g[3],g[4]))


if __name__ == '__main__':
    goods = '小说'
    depth = 4
    count = 0
    start_url = 'https://book.douban.com/tag/' + goods+'?'
    infoList = []
    for i in range(depth):
        try:
            url = start_url + 'start=' + str(20* i)
            html = getHTMLText(url)
            parsePage(infoList, html,goods,count)
            count+=20
        except:
            continue
    printGoodsList(infoList)


