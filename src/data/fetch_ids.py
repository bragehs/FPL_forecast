import requests
import json
import os

def fetch_player_ids():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        print("Error fetching player IDs")
        return {}
    players = data["elements"]
    player_dict = {}
    for player in players:
        player_dict[player["web_name"]] = player["id"]
    return player_dict

if __name__ == "__main__":
    players = fetch_player_ids()
    with open(os.path.join(os.path.dirname(__file__), "player_ids.json"), "w") as f:
        json.dump(players, f)