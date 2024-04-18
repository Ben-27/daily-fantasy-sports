import pandas as pd
import numpy as np
import pulp


def load_fanduel_playerslist(file, all_cols=False):
    '''
    Loads FanDuel players list file as dataframe.
    '''
    ply = pd.read_csv(file)
    # clean and select columns
    ply['FPPG'] = ply['FPPG'].round(2)
    if not all_cols:
        ply = ply[['Id', 'Nickname', 'Position', 'FPPG', 'Salary', 'Game', 'Team', 'Injury Indicator']]
    return ply


def one_hot_positions(df, position_col, inplace=False):
    '''
    Encodes positions into one-hot columns. Mutates df if inplace=True.
    '''
    if not inplace:
        df = df.copy()
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        df[pos] = df[position_col].str.contains(pos).astype(int) 
    return df


def filter_players(df, exclude_injuries=['O'], fppg_cutoff=10, salary_cutoff=4000, verbose=False):
    '''
    Filters out players based on fields from FanDuel file.
        - Injury Indicator
        - FPPG
        - Salary
        
    Does not mutate input dataframe.
    '''
    dat = df.copy()
    
    injury_mask = dat['Injury Indicator'].isin(exclude_injuries)
    fppg_mask = (dat['FPPG'] < fppg_cutoff) | (dat['FPPG'].isna())
    salary_mask = dat['Salary'] < salary_cutoff
    
    full_mask = injury_mask | fppg_mask | salary_mask
    result = dat[~full_mask]
    
    if verbose:
        print(f'Total players to start: {df.shape[0]}')
        print(f'Total players filtered out: {sum(full_mask)}')
        print(f'Total players remaining: {result.shape[0]}')
        print('\nBreakdown by category. Possible duplicates across categories.')
        print(f'Dropping {sum(injury_mask)} players due to injuries.')
        print(f'Dropping {sum(fppg_mask)} players due to low projections.')
        print(f'Dropping {sum(salary_mask)} players due to low salary.')
    
    return result
        
    
def _create_player_dict(df, name_col='Nickname', position_col='Position'):
    '''
    Returns a dictionary of positions for each player.
    Nickname: (Position1, Position2)
    '''
    player_dict = {}
    
    for i in df.index:
        nicknames = df.loc[i, name_col]
        positions = df.loc[i, position_col]
        
        # find position separator
        sep = positions.find('/')
        
        # single position players
        if sep < 0:
            player_dict[nicknames] = (positions,)
            
        # multi position players
        else:
            player_dict[nicknames] = (positions[:sep], positions[sep+1:])
            
    return player_dict


def create_lp_df(df, name_col='Nickname', position_col='Position', add_one_hot=True):
    '''
    Creates a copy of df that includes separate rows for each player's position.
    '''
    df_new = pd.DataFrame(columns=df.columns.to_list() + ['LP Position', 'LP Name'])
    player_dict = _create_player_dict(df, name_col, position_col)
    
    for player in player_dict:
        # each position
        for pos in player_dict[player]:
            temp = df[df[name_col] == player].iloc[0]
            
            # LP labels
            temp['LP Position'] = pos
            temp['LP Name'] = pos + ' ' + player
            
            # append to dataframe
            df_new.loc[df_new.shape[0]] = temp
            
    # create one-hote position columns
    if add_one_hot:
        one_hot_positions(df_new, position_col, inplace=True)
        
    return df_new


def defineProblem(df, point_col, n_studs=0):
    '''
    Linear Program problem definition
    '''
    points = df[point_col]
    consts = df[['PG', 'SG', 'SF', 'PF', 'C', 'Salary', 'Stud']]
    
    # initialize problem
    problem = pulp.LpProblem('Roster', pulp.LpMaximize)
    
    # initalize player variables
    players = np.zeros_like(df['LP Name'])
    
    for i, p in enumerate(df['LP Name']):
        players[i] = pulp.LpVariable(
            p, lowBound = 0, upBound = 1, cat = pulp.LpInteger
        )
    
    # objective function
    problem += pulp.lpSum(players * points)
    
    # constraints
    problem += pulp.lpSum(players * consts.loc[:, 'PG']) == 2, 'PG Constraint'
    problem += pulp.lpSum(players * consts.loc[:, 'SG']) == 2, 'SG Constraint'
    problem += pulp.lpSum(players * consts.loc[:, 'SF']) == 2, 'SF Constraint'
    problem += pulp.lpSum(players * consts.loc[:, 'PF']) == 2, 'PF Constraint'
    problem += pulp.lpSum(players * consts.loc[:, 'C']) == 1, 'C Constraint'
    problem += pulp.lpSum(players) == 9, 'Number of Players' # !!! is this necessary?
    problem += pulp.lpSum(players * consts.loc[:, 'Salary']) <= 60000, 'Salary Constraint'
    problem += pulp.lpSum(players * consts.loc[:, 'Stud']) >= n_studs, 'Stud Constraint'
    
    # maximum signle team constraints
    for team in df['Team'].unique():
        problem += pulp.lpSum(players * (df['Team'] == team)) <= n_studs, f'{team} Constraint'
        
    # player uniqueness constraints
    value_counts = df['Nickname'].value_counts()
    dupe_players = set(value_counts[value_counts > 1].index)
    
    for player in dupe_players:
        problem += pulp.lpSum(players * (df['Nickname'] == player)) <= 1, f'{player} Constraint'
    
    return problem


def solveProblem(df, point_col, n_studs=0):
    '''
    Wrapper to define and solve problem.
    '''
    problem = defineProblem(df, point_col, n_studs = n_studs)
    solution = problem.solve()
    
    if solution != 1:
        print(f'Non-optimal solution in columns {point_col}.')
    
    vars_dict = problem.variablesDict()
    
    # helper function to get values from FD name
    def getValue(x):
        # convert FD name to PuLP name
        key = x.replace(' ', '_').replace('-', '_')
        return vars_dict[key].varValue
    
    return df['LP Name'].apply(getValue)


def runSimulation(df, prefix = 'sim', n_sims = 1, n_studs=2):
    '''
    Wrapper to define and solve problem for each simulation.
    '''
    temp = df.copy()
    
    # solve problem and append to df
    for i in range(n_sims):
        temp[f'roster{i}'] = solveProblem(df, f'{prefix}{i}', n_studs = n_studs)
    
    return temp
        