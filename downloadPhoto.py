# -*- coding:utf-8 -*-
import re
import requests
from bs4 import  BeautifulSoup

def dowmloadPic(html, keyword,i):
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)
    print('找到关键词:' + keyword + '的图片，现在开始下载图片...')

    for each in pic_url:
        print('正在下载第' + str(i) + '张图片，图片地址:' + str(each))
        try:
            pic = requests.get(each, timeout=10)
        except requests.exceptions.ConnectionError:
            print('【错误】当前图片无法下载')
            continue

        dir = 'images/' + keyword + '_' + str(i) + '.jpg'
        fp = open(dir, 'wb')
        fp.write(pic.content)
        fp.close()
        i += 1



if __name__ == '__main__':
    num=1
    keyword='郑秀晶'
    i=1
    kv={'word':keyword,'pn':i}
    for i in range(3):
        url ='https://image.baidu.com/search/index?tn=baiduimage&word='+keyword+'&pn='+str(num)
        result = requests.get(url)
        result.encoding=result.apparent_encoding
        demo=result.text
        dowmloadPic(demo,keyword,num)
        num+=30
