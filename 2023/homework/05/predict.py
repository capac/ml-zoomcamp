import pickle

with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as g_in:
    dv = pickle.load(g_in)

# Transformation
client_dict = {"job": "retired", "duration": 445, "poutcome": "success"}
client = dv.transform(client_dict)

# Prediction
y_pred = model.predict_proba(client)[:, 1]
print(f'Probability of getting a credit: {y_pred[0].round(3)}')
