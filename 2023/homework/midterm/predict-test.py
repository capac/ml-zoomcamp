import requests

# parameters
url = 'http://localhost:9696/predict'

patient_dict = {
    1: {"age":62.0,"anaemia":"No","creatinine_phosphokinase":231,"diabetes":"No","ejection_fraction":25,"high_blood_pressure":"Yes","platelets":253000.0,"serum_creatinine":0.9,"serum_sodium":140,"sex":"Male","smoking":"Yes","time":10},
    2: {"age":65.0,"anaemia":"Yes","creatinine_phosphokinase":128,"diabetes":"Yes","ejection_fraction":30,"high_blood_pressure":"Yes","platelets":297000.0,"serum_creatinine":1.6,"serum_sodium":136,"sex":"Female","smoking":"No","time":20},
    3: {"age":50.0,"anaemia":"Yes","creatinine_phosphokinase":159,"diabetes":"Yes","ejection_fraction":30,"high_blood_pressure":"No","platelets":302000.0,"serum_creatinine":1.2,"serum_sodium":138,"sex":"Female","smoking":"No","time":29}
}

for key, value in patient_dict.items():
    result = requests.post(url, json=value).json()
    if result['outcome']:
        print(f'''Likelihood of death event for patient {str(key)}: Yes''')
    else:
        print(f'''Likelihood of death event for patient {str(key)}: No''')
