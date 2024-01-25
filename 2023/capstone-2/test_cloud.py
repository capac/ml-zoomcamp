import requests
from pprint import pprint

host = "rv4m3mgwb5.execute-api.eu-west-1.amazonaws.com/test"
url = f"https://{host}/predict"

UNSPLASH_URL = "https://images.unsplash.com/photo-1636496430627-a7b203b4cc58"

data = {'url': UNSPLASH_URL}

result = requests.post(url, json=data).json()
pprint(result)
