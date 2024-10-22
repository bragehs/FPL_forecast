from flask import Flask, jsonify
from flask_cors import CORS
from fetch_data import fetch_premier_league_table


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])

def home():
    data = fetch_premier_league_table()
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)