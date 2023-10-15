import requests

# parameters
url = 'http://localhost:9696/predict'

# client for question 2: IMPORTANT - REQUIRES model1.bin
client_1 = {"job": "retired", "duration": 445, "poutcome": "success"}

# client for question 4: IMPORTANT - REQUIRES model1.bin
client_2 = {"job": "unknown", "duration": 270, "poutcome": "success"}

# client for question 6: IMPORTANT - REQUIRES model2.bin
# client_3 = {"job": "retired", "duration": 445, "poutcome": "success"}

client_list = [client_1, client_2]
# client_list = [client_3]

for idx, client in enumerate(client_list):
    result = requests.post(url, json=client).json()
    if result['grant_credit']:
        print(f'''Probability: {round(result['probability'], 3)}''')
        print(f'Client {idx+1} may be granted loan')
