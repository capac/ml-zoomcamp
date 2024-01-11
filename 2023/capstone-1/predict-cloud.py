import requests

host = "zoomcamp-capstone-project-1.onrender.com"
url = f"https://{host}/predict"


patient_dict = {
    1: {
        "age": 62.0,
        "anaemia": "No",
        "creatinine_phosphokinase": 231,
        "diabetes": "No",
        "ejection_fraction": 25,
        "high_blood_pressure": "Yes",
        "platelets": 253000.0,
        "serum_creatinine": 0.9,
        "serum_sodium": 140,
        "sex": "Male",
        "smoking": "Yes",
        "time": 10,
    },
    2: {
        "age": 50.0,
        "anaemia": "Yes",
        "creatinine_phosphokinase": 159,
        "diabetes": "Yes",
        "ejection_fraction": 30,
        "high_blood_pressure": "No",
        "platelets": 302000.0,
        "serum_creatinine": 1.2,
        "serum_sodium": 138,
        "sex": "Female",
        "smoking": "No",
        "time": 29,
    },
}


for key, value in patient_dict.items():
    result = requests.post(url, json=value).json()
    print(f"Patient {key}: {patient_dict[key]}")
    if result["outcome"]:
        print(f"""Outcome of death for patient {str(key)}: Yes.""")
    else:
        print(f"""Outcome of death for patient {str(key)}: No.""")
    print()
