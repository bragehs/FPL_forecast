from flask import Flask, jsonify
from flask_cors import CORS
from scraper import fetch_historical_data

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route('/api/historical', methods=['GET'])
def get_historical_data():
    print("Fetching historical data...")
    data = fetch_historical_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
