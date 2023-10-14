import requests

url = 'http://localhost:9696/predict'

client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
print(requests.post(url, json=client).json())
