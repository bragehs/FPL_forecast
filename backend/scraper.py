from bs4 import BeautifulSoup
import requests
import pandas as pd

def fetch_historical_data():
    # FBref URL for historical Premier League data
    seasons = ['2023-2024', '2022-2023', '2021-2022','2020-2021','2019-2020','2018-2019',
                '2017-2018','2016-2017','2015-2016','2014-2015','2013-2014','2012-2013',
                '2011-2012', '2010-2011']
    
    url = "https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures"
    response = requests.get(url)
    
    if response.status_code != 200:
        return {"error": "Failed to fetch data from FBref"}

    # Parse the page content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing the scores and fixtures
    table = soup.find_all("table")
    print(table)
    if table is None:
        return {"error": "Could not find the scores and fixtures table"}

    # Extract headers from the table
    headers = [th.getText() for th in table.find_all('th')][1:]  # Skip first empty header

    # Extract rows from the table
    rows = table.find_all('tr')
    
    fixtures_data = []
    for row in rows:
        cells = row.find_all(['th', 'td'])
        if len(cells) > 1:
            match_data = [cell.get_text(strip=True) for cell in cells]
            fixture_dict = {headers[i]: match_data[i] for i in range(len(headers))}
            fixtures_data.append(fixture_dict)

    # Save the data to a CSV file for future use
    df = pd.DataFrame(fixtures_data)
    csv_file = "data/2023-2024_premier_league_fixtures.csv"
    df.to_csv(csv_file, index=False)

    return fixtures_data



fetch_historical_data()