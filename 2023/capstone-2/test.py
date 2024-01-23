import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
UNSPLASH_URL = 'https://images.unsplash.com/photo-1636496430627-a7b203b4cc58?q=80&w=2938&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'

data = {'url': UNSPLASH_URL}

result = requests.post(url, json=data).json()
print(result)
