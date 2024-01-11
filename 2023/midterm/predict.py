from flask import Flask, request, jsonify
import pickle

with open('model.pkl', 'rb') as f, open('dv.pkl', 'rb') as g:
    model = pickle.load(f)
    dv = pickle.load(g)

app = Flask('death_event')


@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    X = dv.transform([patient])
    y_pred = model.predict(X)
    result = {
        'outcome': int(y_pred)
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
