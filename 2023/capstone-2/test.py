import requests
from pprint import pprint

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
UNSPLASH_URL = 'https://images.unsplash.com/photo-1636496430627-a7b203b4cc58'

data = {'url': UNSPLASH_URL}

result = requests.post(url, json=data).json()
pprint(result)
