def   getHTMLText(url):
        try :
	r=requests.get(url,timeout=10)
	r.raise_for_status() #若返回值为200则可访问，否则返回错误信息
	r.encoding = r.apparent_encoding
	return r.text
        except :
	return "error"
