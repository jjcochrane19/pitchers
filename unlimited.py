# Code to scrape pitcher strikeout props and the relevant data that I want to test

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
import json
import smtplib, ssl
from lxml import html
import datetime
from datetime import date
import time
import os
import io
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment, PageElement, ResultSet
from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher
from pybaseball import cache
from pybaseball import utils
from pybaseball.utils import most_recent_season, sanitize_date_range
from pybaseball import statcast_pitcher
from pybaseball import pitching_stats
from pybaseball.datasources.fangraphs import fg_team_batting_data
from typing import List, Optional
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from baseball_scraper import pitching_stats_bref

# Authorize the API
scope = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
    ]
file_name = 'freemoney.json'
creds = ServiceAccountCredentials.from_json_keyfile_name(file_name,scope)
client = gspread.authorize(creds)

# Fetch the sheet
worksheet = client.open_by_url('https://docs.google.com/spreadsheets/d/10qq5okYIgb8XchBUVqbWQGZRv_AcuPfnct3Ah0rY0vI/edit#gid=0')
sheet = worksheet.get_worksheet(1)

def next_available_row(worksheet):
    str_list = list(filter(None, worksheet.col_values(1)))
    return str(len(str_list)+1)

pitching_data = (pitching_stats_bref(2023))
for i in range(len(pitching_data)):
    try:
        pitching_data["Tm"][i+1] = str.split(pitching_data["Tm"][i+1], ",")[-1]
        pitching_data = pitching_data.replace({"Tm": di3})
    except:
        pass

# Much of the following code is from the pybaseball repository on Github

team_batting = fg_team_batting_data


@cache.df_cache()
def team_batting_bref(team, start_season, end_season=None):
    """
    Get season-level Batting Statistics for Specific Team (from Baseball-Reference)
    ARGUMENTS:
    team : str : The Team Abbreviation (i.e. 'NYY' for Yankees) of the Team you want data for
    start_season : int : first season you want data for (or the only season if you do not specify an end_season)
    end_season : int : final season you want data for
    """
    if start_season is None:
        raise ValueError(
            "You need to provide at least one season to collect data for. Try team_batting_bref(season) or team_batting_bref(start_season, end_season)."
        )
    if end_season is None:
        end_season = start_season

    url = "https://www.baseball-reference.com/teams/{}".format(team)

    data = []
    headings = None
    for season in range(start_season, end_season+1):
        print("Getting Batting Data: {} {}".format(season, team))
        stats_url = "{}/{}.shtml".format(url, season)
        response = requests.get(stats_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find_all('table', {'class': 'sortable stats_table'})[0]

        if headings is None:
            headings = [row.text.strip() for row in table.find_all('th')[1:28]]

        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            cols = [col.replace('*', '').replace('#', '') for col in cols]  # Removes '*' and '#' from some names
            cols = [col for col in cols if 'Totals' not in col and 'NL teams' not in col and 'AL teams' not in col]  # Removes Team Totals and other rows
            cols.insert(2, season)
            data.append([ele for ele in cols[0:]])

    headings.insert(2, "Year")
    data = pd.DataFrame(data=data, columns=headings) # [:-5]  # -5 to remove Team Totals and other rows
    data = data.dropna()  # Removes Row of All Nones

    return data

def get_soup(year: int) -> BeautifulSoup:
    url = f'http://www.baseball-reference.com/leagues/MLB/{year}-standings.shtml'
    s = requests.get(url).content
    return BeautifulSoup(s, "lxml")

def get_tables(soup: BeautifulSoup, season: int) -> List[pd.DataFrame]:
    datasets = []
    if season >= 1969:
        tables: List[PageElement] = soup.find_all('table')
        if season == 1981:
            # For some reason BRef has 1981 broken down by halves and overall
            # https://www.baseball-reference.com/leagues/MLB/1981-standings.shtml
            tables = [x for x in tables if 'overall' in x.get('id', '')]
        for table in tables:
            data = []
            headings: List[PageElement] = [th.get_text() for th in table.find("tr").find_all("th")]
            data.append(headings)
            table_body: PageElement = table.find('tbody')
            rows: List[PageElement] = table_body.find_all('tr')
            for row in rows:
                cols: List[PageElement] = row.find_all('td')
                cols_text: List[str] = [ele.text.strip() for ele in cols]
                cols_text.insert(0, row.find_all('a')[0].text.strip()) # team name
                data.append([ele for ele in cols_text if ele])
            datasets.append(data)
    else:
        data = []
        table = soup.find('table')
        headings = [th.get_text() for th in table.find("tr").find_all("th")]
        headings[0] = "Name"
        if season >= 1930:
            for _ in range(15):
                headings.pop()
        elif season >= 1876:
            for _ in range(14):
                headings.pop()
        else:
            for _ in range(16):
                headings.pop()
        data.append(headings)
        table_body = table.find('tbody')
        rows = table_body.find_all('tr')
        for row in rows:
            if row.find_all('a') == []:
                continue
            cols = row.find_all('td')
            if season >= 1930:
                for _ in range(15):
                    cols.pop()
            elif season >= 1876:
                for _ in range(14):
                    cols.pop()
            else:
                for _ in range(16):
                    cols.pop()
            cols = [ele.text.strip() for ele in cols]
            cols.insert(0,row.find_all('a')[0].text.strip()) # team name
            data.append([ele for ele in cols if ele])
        datasets.append(data)
    #convert list-of-lists to dataframes
    for idx in range(len(datasets)):
        datasets[idx] = pd.DataFrame(datasets[idx])
    return datasets #returns a list of dataframes


@cache.df_cache()
def standings(season:Optional[int] = None) -> pd.DataFrame:
    # get most recent standings if date not specified
    if season is None:
        season = most_recent_season()
    if season < 1876:
        raise ValueError(
            "This query currently only returns standings until the 1876 season. "
            "Try looking at years from 1876 to present."
        )

    # retrieve html from baseball reference
    soup = get_soup(season)
    if season >= 1969:
        raw_tables = get_tables(soup, season)
    else:
        t = [x for x in soup.find_all(string=lambda text:isinstance(text,Comment)) if 'expanded_standings_overall' in x]
        code = BeautifulSoup(t[0], "lxml")
        raw_tables = get_tables(code, season)
    tables = [pd.DataFrame(table) for table in raw_tables]
    for idx in range(len(tables)):
        tables[idx] = tables[idx].rename(columns=tables[idx].iloc[0])
        tables[idx] = tables[idx].reindex(tables[idx].index.drop(0))
    return tables

def get_soup(year: int) -> BeautifulSoup:
    url = f'http://www.baseball-reference.com/leagues/MLB/{year}-standings.shtml'
    s = requests.get(url).content
    return BeautifulSoup(s, "lxml")

def get_tables(soup: BeautifulSoup, season: int) -> List[pd.DataFrame]:
    datasets = []
    if season >= 1969:
        tables: List[PageElement] = soup.find_all('table')
        if season == 1981:
            # For some reason BRef has 1981 broken down by halves and overall
            # https://www.baseball-reference.com/leagues/MLB/1981-standings.shtml
            tables = [x for x in tables if 'overall' in x.get('id', '')]
        for table in tables:
            data = []
            headings: List[PageElement] = [th.get_text() for th in table.find("tr").find_all("th")]
            data.append(headings)
            table_body: PageElement = table.find('tbody')
            rows: List[PageElement] = table_body.find_all('tr')
            for row in rows:
                cols: List[PageElement] = row.find_all('td')
                cols_text: List[str] = [ele.text.strip() for ele in cols]
                cols_text.insert(0, row.find_all('a')[0].text.strip()) # team name
                data.append([ele for ele in cols_text if ele])
            datasets.append(data)
    else:
        data = []
        table = soup.find('table')
        headings = [th.get_text() for th in table.find("tr").find_all("th")]
        headings[0] = "Name"
        if season >= 1930:
            for _ in range(15):
                headings.pop()
        elif season >= 1876:
            for _ in range(14):
                headings.pop()
        else:
            for _ in range(16):
                headings.pop()
        data.append(headings)
        table_body = table.find('tbody')
        rows = table_body.find_all('tr')
        for row in rows:
            if row.find_all('a') == []:
                continue
            cols = row.find_all('td')
            if season >= 1930:
                for _ in range(15):
                    cols.pop()
            elif season >= 1876:
                for _ in range(14):
                    cols.pop()
            else:
                for _ in range(16):
                    cols.pop()
            cols = [ele.text.strip() for ele in cols]
            cols.insert(0,row.find_all('a')[0].text.strip()) # team name
            data.append([ele for ele in cols if ele])
        datasets.append(data)
    #convert list-of-lists to dataframes
    for idx in range(len(datasets)):
        datasets[idx] = pd.DataFrame(datasets[idx])
    return datasets #returns a list of dataframes


@cache.df_cache()
def standings(season:Optional[int] = None) -> pd.DataFrame:
    # get most recent standings if date not specified
    if season is None:
        season = most_recent_season()
    if season < 1876:
        raise ValueError(
            "This query currently only returns standings until the 1876 season. "
            "Try looking at years from 1876 to present."
        )

    # retrieve html from baseball reference
    soup = get_soup(season)
    if season >= 1969:
        raw_tables = get_tables(soup, season)
    else:
        t = [x for x in soup.find_all(string=lambda text:isinstance(text,Comment)) if 'expanded_standings_overall' in x]
        code = BeautifulSoup(t[0], "lxml")
        raw_tables = get_tables(code, season)
    tables = [pd.DataFrame(table) for table in raw_tables]
    for idx in range(len(tables)):
        tables[idx] = tables[idx].rename(columns=tables[idx].iloc[0])
        tables[idx] = tables[idx].reindex(tables[idx].index.drop(0))
    return tables

# My code again

url = 'https://www.rotowire.com/betting/mlb/player-props.php'
soup = BeautifulSoup(requests.get(url).content, 'html.parser')

script_tag = soup.find(lambda tag: tag.name == 'script' and tag.text and '"Strikeouts"' in tag.text)

data = re.search(r"data:\s*(\[.*\])", script_tag.text)
data = json.loads(data.group(1))

data_frame = pd.DataFrame(data)

games = "https://www.fantasypros.com/mlb/schedules/"
page = requests.get(games)
games_frame = pd.read_html(page.text, displayed_only=False)[0]
di = {"Diamondbacks":"ARI",
      
"Braves": "ATL",

"Orioles": "BAL",

"Red Sox":"BOS",

"Cubs":"CHC",

"White Sox":"CHA",

"Reds": "CIN",

"Guardians": "CLE",

"Rockies": "COL",

"Tigers": "DET",

"Marlins": "MIA",

"Astros": "HOU",

"Royals": "KC",

"Angels": "LAA",

"Dodgers": "LAD",

"Brewers": "MIL",

"Twins": "MIN",

"Mets": "NYM",

"Yankees": "NYY",

"Athletics": "OAK",

"Phillies": "PHI",

"Pirates": "PIT",

"Padres": "SD",

"Giants": "SF",

"Mariners": "SEA",

"Cardinals": "STL",

"Rays": "TB",

"Rangers": "TEX",

"Blue Jays": "TOR",

"Nationals": "WAS"}

di2 = {"Arizona Diamondbacks":"ARI",
      
"Atlanta Braves": "ATL",

"Baltimore Orioles": "BAL",

"Boston Red Sox":"BOS",

"Chicago Cubs":"CHC",

"Chicago White Sox":"CHA",

"Cincinnati Reds": "CIN",

"Cleveland Guardians": "CLE",

"Colorado Rockies": "COL",

"Detroit Tigers": "DET",

"Miami Marlins": "MIA",

"Houston Astros": "HOU",

"Kansas City Royals": "KC",

"Los Angeles Angels": "LAA",

"Los Angeles Dodgers": "LAD",

"Milwaukee Brewers": "MIL",

"Minnesota Twins": "MIN",

"New York Mets": "NYM",

"New York Yankees": "NYY",

"Oakland Athletics": "OAK",

"Philadelphia Phillies": "PHI",

"Pittsburgh Pirates": "PIT",

"San Diego Padres": "SD",

"San Francisco Giants": "SF",

"Seattle Mariners": "SEA",

"St. Louis Cardinals": "STL",

"Tampa Bay Rays": "TB",

"Texas Rangers": "TEX",

"Toronto Blue Jays": "TOR",

"Washington Nationals": "WAS"}

di3 = {"Arizona":"ARI",
      
"Atlanta": "ATL",

"Baltimore": "BAL",

"Boston":"BOS",

"Chicago":"CHC",

"Chicago":"CHA",

"Cincinnati": "CIN",

"Cleveland": "CLE",

"Colorado": "COL",

"Detroit": "DET",

"Miami": "MIA",

"Houston": "HOU",

"Kansas City": "KC",

"Los Angeles": "LAA",

"Los Angeles": "LAD",

"Milwaukee": "MIL",

"Minnesota": "MIN",

"New York": "NYM",

"New York": "NYY",

"Oakland": "OAK",

"Philadelphia": "PHI",

"Pittsburgh": "PIT",

"San Diego": "SD",

"San Francisco": "SF",

"Seattle": "SEA",

"St. Louis": "STL",

"Tampa Bay": "TB",

"Texas": "TEX",

"Toronto": "TOR",

"Washington": "WAS"}

games_frame.columns = ['Away', 'Home', '3', "4", '5', '6', '7', '8']
games_frame = games_frame.replace({"Away": di})
games_frame = games_frame.replace({"Home": di})
games = games_frame[["Home", "Away"]].dropna()
a = games[games['Home'].str.contains(',')]

end_int = a.index.values.tolist()[0] - 1

games = games.loc[:end_int]

dict1 = pd.Series(games.Home.values,index=games.Away).to_dict()
dict2 = dict((v,k) for k,v in dict1.items())
games_dict = {**dict1, **dict2}
standings = standings()

standings = pd.concat(standings).replace({"Tm": di2}).reset_index()

def update_game(first_name, last_name, team, ou):
    
    alt_name = None
    if team == "CWS":
        team = "CHA"
    else:
        pass

    opp = games_dict[team]
        
    if opp == "CWS":
        opp = "CHA"
        alt_name = "CWS"
    if opp == "WAS":
        opp = "WSN"
        alt_name = "WAS"
    else:
        pass
   
    batting_data = team_batting_bref(opp, 2023)
    batting_data = batting_data[batting_data['Pos'] != "P"]
    print(batting_data)
    index = next_row = next_available_row(sheet)
    try:
        pitcher_data_local = pitching_data[pitching_data['Name'] == (first_name + " " + last_name)]
        player_info = playerid_lookup(last_name, first_name)
        sheet.update_cell(index,1,str(date.today()))
        sheet.update_cell(index,2,first_name+" "+last_name)
        first_date = '2023-01-01'
        last_date = '2023-12-12'
        player_key = str(player_info["key_mlbam"][0])
        sheet.update_cell(index, 3, ou)
        sheet.update_cell(index, 4, (statcast_pitcher(first_date, last_date, player_key)["release_speed"]).mean())
        sheet.update_cell(index, 5, (statcast_pitcher(first_date, last_date, player_key)["release_pos_x"]).mean())
        sheet.update_cell(index, 6, (statcast_pitcher(first_date, last_date, player_key)["release_pos_y"]).mean())
        sheet.update_cell(index, 7, (statcast_pitcher(first_date, last_date, player_key)["release_pos_z"]).mean())
        sheet.update_cell(index, 8, float(pitcher_data_local["SO9"]))
        sheet.update_cell(index, 9, float(pitcher_data_local["SO/W"]))
        sheet.update_cell(index, 10, float(pitcher_data_local["WHIP"]))
        sheet.update_cell(index, 11, float(pitcher_data_local["ERA"]))
        sheet.update_cell(index, 12, float(pitcher_data_local["GB/FB"]))
        sheet.update_cell(index, 13, opp)
        opp_info = standings.loc[standings["Tm"] == opp]
        opp_idx = list(opp_info.index.values)[0]
        if opp == "WSN":
            win_perc = float(standings.loc[standings['Tm'] == "WAS"]["W-L%"][opp_idx])
        else:
            win_perc = float(standings.loc[standings['Tm'] == opp]["W-L%"][opp_idx])
        
        sheet.update_cell(index, 14, win_perc)
        sheet.update_cell(index, 15, sum(list(map(int,list(batting_data["H"]))))/sum(list(map(int,list(batting_data["AB"])))))
        sheet.update_cell(index, 16, (sum(list(map(int,list(batting_data["H"])))) + sum(list(map(int,list(batting_data["HBP"])))) +
                                     sum(list(map(int,list(batting_data["BB"]))))) / sum(list(map(int,list(batting_data["PA"]))))
                          )
        pa_list = list((list(map(int,list(batting_data["PA"][0:22])))))
        try:
            ops_plus_list = list((list(map(int,list(batting_data["OPS+"][0:22])))))
        except:
            pass
        multiplied = sum(list(np.multiply(pa_list, ops_plus_list)))
        
        sheet.update_cell(index, 17, multiplied / (sum(list(map(int,list(batting_data["PA"]))))))
        sheet.update_cell(index, 18, sum(list(map(int,list(batting_data["SO"])))))
        sheet.update_cell(index, 19, sum(list(map(int,list(batting_data["BB"])))))
    except:
        pass

for i in range(len(data_frame)):
    first_name = data_frame["firstName"][i]
    last_name = data_frame["lastName"][i]
    team = data_frame["team"][i]
    matchup = data_frame["opp"][i]
    ou = data_frame["fanduel_strikeouts"][i]
    update_game(first_name, last_name, team, ou)
    time.sleep(20)
