import sys,re,html,urllib.request,time
ids=sys.argv[1].split(',')
for k in range(0,len(ids),20):
    u='https://export.arxiv.org/api/query?max_results=50&id_list='+','.join(ids[k:k+20])
    for a in range(4):
        try: x=urllib.request.urlopen(u,timeout=60).read().decode(); break
        except Exception: time.sleep(5*(a+1))
    for e in x.split('<entry>')[1:]:
        g=lambda t: html.unescape(re.sub(r'\s+',' ',(re.search(f'<{t}>(.*?)</{t}>',e,re.S) or [None,''])[1])).strip()
        i=re.search(r'<id>http://arxiv.org/abs/(.*?)</id>',e).group(1)
        c=re.search(r'<arxiv:comment[^>]*>(.*?)</arxiv:comment>',e,re.S)
        print(f"### {i} | {g('published')[:10]} | {g('title')}\n{('[comment] '+html.unescape(c.group(1).strip())) if c else ''}\n{g('summary')}\n")
    time.sleep(3)
