import sys, urllib.request, urllib.parse, re, time, html
q=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 50
url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':q,'start':0,'max_results':n,'sortBy':'relevance'})
for attempt in range(4):
    try:
        x=urllib.request.urlopen(url,timeout=60).read().decode(); break
    except Exception as e:
        time.sleep(5*(attempt+1))
ents=x.split('<entry>')[1:]
print(f'## {q} -> {len(ents)}')
for e in ents:
    t=html.unescape(re.sub(r'\s+',' ',re.search(r'<title>(.*?)</title>',e,re.S).group(1)))
    i=re.search(r'<id>http://arxiv.org/abs/(.*?)</id>',e).group(1)
    d=re.search(r'<published>(.*?)</published>',e).group(1)[:10]
    print(f'{i} | {d} | {t}')
