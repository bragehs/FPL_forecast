import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_premier_league_table():
    url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve data: {response.status_code}")
        return
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'results2024-202591_overall'})

    if not table:
        print("Failed to find the Premier League table.")
        return
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]

    rows = []

    for row in table.find('tbody').find_all('tr'):
        ranking = row.find('th').text.strip()
        cells = row.find_all('td')
        row_data = {headers[0]: ranking}
        row_data.update({headers[i+1]: cells[i].text.strip() for i in range(len(cells))})
        rows.append(row_data)
    return rows

if __name__ == "__main__":
    teams = fetch_premier_league_table()