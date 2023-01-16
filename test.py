import urllib

req = urllib.request.Request('https://v2.gcchmc.org/book-appointment/')
req.add_headers('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0')
req.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8')
req.add_header('Accept-Language', 'en-US,en;q=0.5')

r = urllib.request.urlopen(req).read().decode('utf-8')
with open("test.html", 'w', encoding="utf-8") as f:
    f.write(r)