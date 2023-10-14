from flask import Flask, request, jsonify
import pickle

with open('model1.bin', 'rb') as f_in, open('dv.bin', 'rb') as g_in:
    model = pickle.load(f_in)
    dv = pickle.load(g_in)


app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    results = {
        'probability': float(y_pred)
    }
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
