from flask import Flask, request, jsonify
import pickle

with open('model1.bin', 'rb') as f_in, open('model2.bin', 'rb') as g_in:
    model1 = pickle.load(f_in)
    model2 = pickle.load(g_in)

with open('dv.bin', 'rb') as h_in:
    dv = pickle.load(h_in)

app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model1.predict_proba(X)[0, 1]
    grant_credit = y_pred > 0.5
    results = {
        'probability': float(y_pred),
        'grant_credit': bool(grant_credit)
    }
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
