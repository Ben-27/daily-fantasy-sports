import pandas as pd
import requests
from bs4 import BeautifulSoup

def _extract_data_from_div(d):
    """
    Returns list of data points
    """
    position = d['data-position']
    team = d['data-team']
    first = d['data-first-name']
    last = d['data-last-name']
    # convert full suffix to player-specific
    suffix = d.find('a', href=True)['href']
    suffix = suffix[suffix.find('players/')+8:]
    
    return [first, last, position, team, suffix]


def _build_df(data):
    """
    Build dataframe by iterating through player list
    """
    df = pd.DataFrame(columns=['first', 'last', 'position', 'team', 'url_suffix'])
    for d in data:
        df.loc[df.shape[0]] = _extract_data_from_div(d)
    return df


def download_nf_players():
    """
    Downloads players from numberfire.com/nba/players
    
    Returns dataframe: first, last, position, team, url_suffix
    """
    page = requests.get('https://www.numberfire.com/nba/players')
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # find data via class attribute
    plys = soup.find_all(class_='all-players__indiv')
    return _build_df(plys)


def get_game_log(url_suffix):
    """
    Returns dict of tables from numberfire.com/nba/players/daily-fantasy/<player-suffix>
    
    Keys: 'past', 'upcoming'
    """
    url = 'https://www.numberfire.com/nba/players/daily-fantasy/' + url_suffix
    tables = pd.read_html(url)
    
    # past and upcoming are present
    if len(tables)==4:    
        return {
            'past': pd.concat([tables[2], tables[3]], axis='columns'),
            'upcoming': pd.concat([tables[0], tables[1]], axis='columns')
        }
    
    # past only
    else:
        return {
            'past': pd.concat([tables[0], tables[1]], axis='columns')
        }
    
    
def concatenate_logs(urls):
    """
    Gets game logs for all players and concatenates tables
    
    Returns dict of dataframes
    Keys: 'past', 'upcoming'
    """
    past_logs = []
    upcoming_logs = []
    errored_urls = []
    
    for i, url_suffix in enumerate(urls):
        try:
            logs = get_game_log(url_suffix)

            # add identifying column to each
            logs['past']['url_suffix'] = url_suffix
            past_logs.append(logs['past'])
            
            if 'upcoming' in logs.keys():
                logs['upcoming']['url_suffix'] = url_suffix
                upcoming_logs.append(logs['upcoming'])
        
        except:
            print(f'Error with {i}, {url_suffix}')
            errored_urls.append(url_suffix)

        # status bar
        print(f"{(i+1)/len(urls):6.1%} completed\r", end="")
        
    return {'past': pd.concat(past_logs), 'upcoming': pd.concat(upcoming_logs), 'errors': errored_urls}


def scrape_all(file_name):
    """
    Single call to save all data.
    """
    plys = download_nf_players()
    print(f'{plys.shape[0]} players found.')
    logs = concatenate_logs(plys.loc[:, 'url_suffix'])
    
    with pd.ExcelWriter(file_name) as writer:
        plys.to_excel(writer, sheet_name='players', index=False)
        logs['past'].to_excel(writer, sheet_name='past', index=False)
        logs['upcoming'].to_excel(writer, sheet_name='upcoming', index=False)