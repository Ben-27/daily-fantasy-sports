"""
Daily Fantasy Sports

@author: Benjamin Absalon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp
import os
from itertools import chain, combinations
from datetime import datetime


class Game:
    """
    Object to hold and process all data and attributes for a FanDuel game. 
    
    Loading a saved game:
        base_file : file name in Lib/Saved Games/ without extension
        sport : sport
        
    Creating a new game:
        base_file : FanDuel player list
        sport : sport
        projections : file name in Lib/Projections/ with extension
        historical : file name in Lib/Historical Stats/ with extension
        season : four digit integer representing year
        week : week number
        
    Optional:
        n_historical : number of games to summarize in actuals
        name_mapping : translation of nicknames
    """
    
    def __init__(self, base_file, sport,
                 projections=None, historical=None, season=None, week=None, 
                 n_historical=16, name_mapping=None):
        
        if projections:
            assert historical and season and week and n_historical
            """
            Initialize based on data files.            
            Used to create games from scratch.
            """
            # set attributes
            self._players_list = base_file
            self._game_id = int(base_file[
                base_file.find('-players')-5 : base_file.find('-players')])
            self._projections = projections
            self._historical = historical
            self._sport = sport
            self._season = season
            self._week = week
            self._n_historical = n_historical
            if name_mapping:
                self._name_mapping = name_mapping
            else:
                self._name_mapping = self.getDefaultNameMapping()
            
            # load projections, players list, historical, and actuals dataframes
            self.projections = self.loadFantasyProsProjections()
            pl = self.loadFanDuelPlayersList()
            self.players_list = pl
            historical, actuals = self.loadFantasyProsActuals()
            self.historical = historical
            self.actuals = actuals
            # merge players list with projections and summarized actuals
            pl_m = pl.merge(
                self.projections, how = 'left', left_on = 'Nickname', right_on = 'Nickname')
            self.data = pl_m.merge(
                self.summarizeActuals(), how = 'left', left_on = 'Nickname', right_on = 'Nickname')
            # initialize rosters
            self.rosters = pd.DataFrame(
                columns=['Salary', 'Projected', 'Hist_Avg', 'Hist_Std', 'Proj_Percent', 'K'])
            
            # initialize tracking dict
            self.tracking = {}
        
        else:
            """
            Initialize based on game_file that contains meta data and 
            master dataframe.
            Used to analyze past games.
            """
            # load meta, data, and rosters
            self.import_game(base_file, sport)
            
            # load historical, acutal
            historical, actuals = self.loadFantasyProsActuals()
            self.historical = historical
            self.actuals = actuals
            
            # merge actuals to data and rosters
            self.mergeActuals()
            
            # initialize tracking dict
            self.tracking = {}
        
        
    # getters and setters -----------------------------------------------------
    
    def getDefaultNameMapping(self):
        """
        Mapping of Fantasy Pros' player names to FanDuel names.
        """
        if self._sport == 'NFL':
            return {'Elijah Mitchell': 'Eli Mitchell',
                     'Patrick Mahomes II': 'Patrick Mahomes',
                     'Duke Johnson Jr.': 'Duke Johnson',
                     'PJ Walker': 'P.J. Walker',
                     'Patrick Taylor Jr.': 'Patrick Taylor',
                     'Keelan Cole Sr.': 'Keelan Cole',
                     'Gabriel Davis': 'Gabe Davis',
                     'Demetric Felton Jr.': 'Demetric Felton',
                     'Ray-Ray McCloud': 'Ray-Ray McCloud III',
                     'Stanley Morgan Jr.': 'Stanley Morgan'}
        else:
            raise NotImplementedError('Default name mapping not implemented.')
    
    
    # FILE FUNCTIONS ----------------------------------------------------------
    
    def loadFanDuelPlayersList(self, full=False):
        """
        Load players list with binary position columns.
        """
        folder = f'Lib/Players Lists/{self._sport}/'
        fd = pd.read_csv(folder + self._players_list)        
        # create binary position columns
        if self._sport == 'NFL':
            fd['QB'] = (fd['Position'] == 'QB')*1
            fd['RB'] = (fd['Position'] == 'RB')*1
            fd['WR'] = (fd['Position'] == 'WR')*1
            fd['TE'] = (fd['Position'] == 'TE')*1
            fd['DST'] = (fd['Position'] == 'D')*1
            fd['FLEX'] = (fd['Position'].isin(['RB', 'WR', 'TE']))*1
            # return full or pared down dataframe
            if full:
                return fd
            return fd[['Nickname', 'Position', 'Salary', 'Injury Indicator',
                       'QB', 'RB', 'WR', 'TE', 'DST', 'FLEX']]
        
        if self._sport == 'NBA':
            # TODO
            raise NotImplementedError()
        
        
    def loadFantasyProsProjections(self, full=False):
        """
        Load projections.
        """
        folder = f'Lib/Projections/{self._sport}/'
        fd = pd.read_excel(folder + self._projections)
        
        # clean names and map to FanDuel
        fd['Nickname'] = fd['Nickname'].str.replace('\xa0', '').apply(
            lambda x: self._name_mapping.get(x, x))
        
        fd.columns = ['Projected' if x=='FPTS' else x for x in fd.columns]
        if full:
            return fd
        return fd[['Nickname', 'Projected']]
    
    
    def loadFantasyProsActuals(self, filter_G=True):
        """
        Load actuals for historical and weekly.
        
        Return tuple containing (historical, actual)
        """
        folder = f'Lib/Historical Stats/{self._sport}/'
        if self._sport == 'NFL':
            positions = ['qb', 'rb', 'wr', 'te', 'dst']
        else:
            # TODO implement NBA
            raise NotImplementedError()
        
        # load each sheet in actuals
        dat = []
        for pos in positions:
            temp = pd.read_excel(folder + self._historical, sheet_name=pos,
                                 converters={'ROST' : lambda x: float(x[:-1])})
            dat.append(temp[['Player', 'FPTS', 'Season', 'WeekNum', 'G', 'ROST']])
            
        # concatenate data into one dataframe and filter to G>0
        df = pd.concat(dat)
        if filter_G:
            df = df[df['G'] > 0]
        
        # add team, nickname, and period columns
        df['Nickname'] = df['Player'].apply(lambda x: x[:x.find('(')-1])
        df['Period'] = df['Season'].astype(str) + '-' + df['WeekNum'].astype(str).str.zfill(2)
        
        # sort
        df.sort_values('Period', ascending=False, inplace=True)
        
        # map nicknames FantasyPros -> FanDuel
        df['Nickname'] = df['Nickname'].apply(lambda x: self._name_mapping.get(x,x))
        
        # filter actuals
        current_period = f'{self._season}-{str(self._week).zfill(2)}'
        historical_mask = (df['Period'] < current_period)
        weekly_mask = (df['Period'] == current_period)
        
        # return historical, actual
        return df[historical_mask], df[weekly_mask]
    
    
    def summarizeActuals(self):
        """
        Summarize actuals with n most recent games.
        """
        # summarize historical stats and merge with players list to create data
        act = self.historical.copy().sort_values('Period', ascending=False)[
            ['Nickname', 'Period', 'Season', 'WeekNum', 'G', 'FPTS', 'FPTS', 'ROST']]
        
        # grab n most recent games
        top = act.groupby('Nickname').head(self._n_historical)
        
        # prep columns, aggregate, and clean up
        top.columns = ['Nickname', 'MR_Period', 'MR_Season', 'MR_WeekNum',
                       'N_Games', 'Hist_Avg', 'Hist_Std', 'Proj_Percent']
        agg = top.groupby('Nickname').aggregate(
            {'MR_Period':'first', 'MR_Season':'first', 'MR_WeekNum':'first',
             'N_Games':'sum', 'Hist_Avg':'mean', 'Hist_Std':'std', 'Proj_Percent':'first'})
        agg['Hist_Avg'] = round(agg['Hist_Avg'], 2)
        agg['Hist_Std'] = round(agg['Hist_Std'], 2)
        return agg
    
    
    def mergeActuals(self):
        """
        Merge actual fantasy points into data.
        """
        # merge points onto data
        self.data = self.data.merge(
            self.actuals[['Nickname', 'FPTS']],
            how = 'left', left_on = 'Nickname', right_on='Nickname')
        # rename 'FPTS' to 'Actual'
        self.data.rename(columns={'FPTS' : 'Actual'}, inplace=True)
        
        # merge actual roster points to rosters dataframe
        rosters = self.data[
            self.data.columns[list(self.data.columns).index('Roster0') : -1]
            ].replace(-1, 0)
        points = self.data['Actual']
        r_points = rosters.multiply(points.fillna(0), axis='rows').sum(axis='rows')
                    
        self.rosters['Actual'] = r_points
    
    
    def import_game(self, file_name, sport):
        """
        Load meta and roster data from Excel game file, data from Pickle. 
        Set attributes.
        """
        # truncate file type if needed
        if file_name.find('.xlsx') >= 0:
            raise ValueError(f'{file_name} must not contain extension ".xlsx"')
        
        full_file_name = f'Lib/Saved Games/{sport}/{file_name}'
        # load data and set attributes
        self.data = pd.read_pickle(full_file_name)
        
        # load meta and set attributes
        meta = pd.read_excel(full_file_name + '.xlsx',
                             sheet_name = 'meta',
                             index_col = 0)
        attributes = {
            '_sport' : 'Sport',
            '_season' : 'Season',
            '_week' : 'Week',
            '_game_id' : 'GameID',
            '_players_list' : 'PlayersList',
            '_projections' : 'Projections',
            '_historical' : 'Historical',
            '_actuals' : 'Actuals',
            '_strategy' : 'Strategy',
            '_selection' : 'Selection',
            '_salary_limit' : 'SalaryLimit',
            '_min_points' : 'MinPoints',
            '_q_points' :'q_points',
            '_q_std' : 'q_std',
            '_n_historical' : 'n_historical'
            }
        for a in attributes.keys():
            try:
                value = meta.loc[attributes[a]][0]
                # convert nan to None
                if pd.isna(value):
                    setattr(self, a, None)
                else:
                    setattr(self, a, value)
            except:
                setattr(self, a, None)
        # try to assign drop players and injuries
        try:
            self._drop_players = meta.loc['DropPlayers'][0].split(',')
        except:
            self._drop_players = None
        try:
            self._drop_injuries = meta.loc['DropInjuries'][0].split(',')
        except:
            self._drop_injuries = None
        
        # name mapping helper
        self._name_mapping = {}
        meta_k = meta.loc['NameMapKeys'][0].split(',')
        meta_v = meta.loc['NameMapValues'][0].split(',')
        for i, e in enumerate(meta_k):
            self._name_mapping[e] = meta_v[i]
        
        # load roster summaries
        try: # TODO clean up try/except
            self.rosters = pd.read_excel(full_file_name + '.xlsx',
                                         sheet_name = 'rosters',
                                         index_col = 0)
        except:
            print('Unable to load rosters.')
        
        return True
    
    
    def export_game(self, file_name=None, override=False):
        """
        Export meta, data, and roster data to Excel file and Pickle file.
        """
        save_folder = f'Lib/Saved Games/{self._sport}/'
        if not file_name:
            file_name = f'GAME {self._sport} {self._season}-' + \
                f'{str(self._week).zfill(2)} {self._game_id} {self._strategy}'
        
        elif file_name.find('.xlsx') >= 0:
            raise ValueError(f'{file_name} should not contain extension ".xlsx"')
        
        def list_to_string(l):
            """
            Helper function to convert dictionary to string.
            """
            x = ''
            for i, e in enumerate(l):
                if i == 0:
                    x = e
                else:
                    x += ',' + e
            return x
        
        def truncate_file(f):
            """
            Helper function to remove file type from file name.
            """
            # assumes two slashes
            return f[f.find('/', f.find('/')+1)+1 :]
        
        # prep meta data
        name_map_keys = None
        name_map_values = None
        if self._name_mapping:
            name_map_keys = list_to_string(self._name_mapping.keys())
            name_map_values = list_to_string(self._name_mapping.values())
            
        drop_players = None
        if self._drop_players:
            drop_players = list_to_string(self._drop_players)
            
        drop_injuries = None
        if self._drop_injuries:
            drop_injuries = list_to_string(self._drop_injuries)
            
        self._actuals = f'{self._season}-{str(self._week).zfill(2)} FP Actuals.xlsx'
    
        meta_data = {
            'Sport' : self._sport,
            'Season' : self._season,
            'Week' : self._week,
            'GameID' : self._game_id,
            'PlayersList': truncate_file(self._players_list),
            'Projections' : truncate_file(self._projections),
            'Historical' : truncate_file(self._historical), 
            'Actuals' : truncate_file(self._actuals),
            'Strategy' : self._strategy,
            'Selection' : self._selection,
            'NameMapKeys' : name_map_keys,
            'NameMapValues' : name_map_values,
            'SalaryLimit' : self._salary_limit,
            'DropPlayers' : drop_players,
            'DropInjuries' : drop_injuries,
            'MinPoints' : self._min_points,
            'q_points' : self._q_points,
            'q_std' : self._q_std,
            'n_historical' : self._n_historical
        }
        meta = pd.DataFrame(meta_data.values(),
                            index = meta_data.keys(),
                            columns=['Values'])
        
        # check if file already exists        
        if (not override) and \
            (os.path.exists(save_folder + file_name) or \
            os.path.exists(save_folder + file_name + '.xlsx')):
            print(f'"{save_folder}{file_name}" already exists!')
            return False
        else:
            data = self.data
            rosters = self.rosters
            
            # truncate actuals if they exist
            if 'Actual' in data.columns:
                data = data[data.columns[:-1]]
                rosters = rosters[rosters.columns[:-1]]
            # save meta, data, and rosters to Excel
            with pd.ExcelWriter(save_folder + file_name + '.xlsx') as writer:
                meta.to_excel(writer, sheet_name = 'meta')
                data.to_excel(writer, sheet_name = 'data')
                rosters.to_excel(writer, sheet_name = 'rosters')
            # save data as pickle file
            data.to_pickle(save_folder + file_name)
        return True
    
    
    def FanDuelCSVExport(self, rosters, save_to_file=True, file_name=None):
        """
        Creates a dataframe in format of FanDuel's CSV import with player IDs.
        Saves file by default.
        
        rosters : list of roster names to export
        """
        
        if type(rosters) != list:
            raise ValueError(f'Rosters is expected to be a list. Got {type(rosters)}.')
        
        folder = 'Lib/FanDuel CSV Imports/'
        if not file_name:
            file_name = f'FanDuel {self._sport} {self._game_id} upload file.csv'
        
        # merge IDs from players list
        pl = pd.read_csv(f'Lib/Players Lists/{self._sport}/' + self._players_list)
        dat = self.data.merge(pl[['Nickname', 'Id']],
                              how = 'left', left_on = 'Nickname', right_on = 'Nickname')        
        
        # create dictionary
        upload = {}
        
        if self._sport == 'NFL':
            for roster in rosters:
                ids = {}
                temp = dat[dat[roster]==1]
                
                ids['QB'] = temp.loc[temp.Position == 'QB', 'Id'].iloc[0]
                
                rbs = temp.loc[temp.Position == 'RB', 'Id']
                ids['RB1'] = rbs.iloc[0]
                ids['RB2'] = rbs.iloc[1]
                if len(rbs) == 3:
                    ids['FLEX'] = rbs.iloc[2]
                    
                wrs = temp.loc[temp.Position == 'WR', 'Id']
                ids['WR1'] = wrs.iloc[0]
                ids['WR2'] = wrs.iloc[1]
                ids['WR3'] = wrs.iloc[2]
                if len(wrs) == 4:
                    ids['FLEX'] = wrs.iloc[3]
                    
                tes = temp.loc[temp.Position == 'TE', 'Id']
                ids['TE'] = tes.iloc[0]
                if len(tes) == 2:
                    ids['FLEX'] = tes.iloc[1]
                    
                ids['DEF'] = temp.loc[temp.Position == 'D', 'Id'].iloc[0]
                
                upload[roster] = ids
        
        else:
            raise NotImplementedError(f'FanDuelExportCSV not yet implemented for {self._sport}')
            
        # conver to dataframe
        df = pd.DataFrame(upload).T
        
        # reorder and rename columns to match FanDuel
        if self._sport == 'NFL':
            df = df[['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DEF']]
            df.columns = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DEF']
        
        # save to file
        if save_to_file:
            print(f'Saving file {folder + file_name}')
            df.to_csv(folder + file_name, index=False)
            
        return df
        
    
    # ROSTER FUNCTIONS --------------------------------------------------------
    
    def clearRosters(self):
        """
        Clears roster columns from data and reinitializes roster dataframe.
        """ 
        dat = self.data
        if 'Roster0' not in dat.columns:
            return True
        
        # remove roster data
        self.data = dat[dat.columns[: list(dat.columns).index('Roster0')]]
        self.rosters = pd.DataFrame(
            columns=['Salary', 'Projected', 'Hist_Avg', 'Hist_Std', 'Proj_Percent', 'K'])
        return True
    
    
    def _powerset(self, s):
        """
        Return a list of all player combinations from s.
        """
        power_s = []
        c = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        for x in c:
            power_s.append(x)
        return power_s[1:]
    
    
    def _defineProblem(self, dat, point_col='Projected', drop_players=[],
                       variance_type=None, variance_target=None, max_te=2):
            """
            Breaks dat into components for integer programming problem.
            Returns LpProblem
            """
            # TODO implement variance options for efficient frontier
            
            if self._sport == 'NFL':
                # break feed into components
                c = dat[point_col]
                A = dat[['QB', 'RB', 'WR', 'TE', 'DST', 'FLEX', 'Salary']].T
    
                # intialize problem
                problem = pulp.LpProblem('Roster', pulp.LpMaximize)
                
                # initialize players variables
                players = np.zeros_like(dat['Nickname'])
                for i, p in enumerate(dat['Nickname']):
                    # set upper limit of dropped players to zero
                    if drop_players and p in drop_players:
                        upper = 0
                    else:
                        upper = 1
                    lower = 0
                    # add players as integer variables
                    players[i] = pulp.LpVariable(p, lowBound=lower, upBound=upper, cat=pulp.LpInteger)
                    
                # objective function
                problem += pulp.lpSum(players * c), 'Objective Function'
                # constraints
                problem += pulp.lpSum(players * A.iloc[0]) == 1, 'QB Constraint'
                problem += pulp.lpSum(players * A.iloc[1]) >= 2, 'RB Min Contraint'
                problem += pulp.lpSum(players * A.iloc[1]) <= 3, 'RB Max Contraint'
                problem += pulp.lpSum(players * A.iloc[2]) >= 3, 'WR Min Contraint'
                problem += pulp.lpSum(players * A.iloc[2]) <= 4, 'WR Max Contraint'
                problem += pulp.lpSum(players * A.iloc[3]) >= 1, 'TE Min Contraint'
                problem += pulp.lpSum(players * A.iloc[3]) <= max_te, 'TE Max Contraint'
                problem += pulp.lpSum(players * A.iloc[4]) == 1, 'DST Contraint'
                problem += pulp.lpSum(players * A.iloc[5]) == 7, 'FLEX Contraint'
                problem += pulp.lpSum(players * A.iloc[6]) <= self._salary_limit, 'Salary Contraint'
                return problem
            
            # TODO implement NBA
            if self._sport == 'NBA':
                raise NotImplementedError('_defineProblem for NBA not yet implemented.')
            
            return False
       
        
    def _solveProblem(self, point_col, roster_name='Roster0', drop_players=[], max_te=2):
        """
        Creates and solves LpProblem from data and appends the roster to the dataframe.
    
        Modifies: self.data
        
        Returns: solution status
        """
        problem = self._defineProblem(self.data, point_col, drop_players, max_te=max_te)
        problem.solve()
        
        vars_dict = problem.variablesDict()
        def getValue(x):
            if drop_players and x in drop_players:
                return -1.0
            else:
                # replace spaces and dashes in Nickname with underscores to match pulp
                return vars_dict[x.replace(' ', '_').replace('-', '_')].varValue
        
        # create roster column and append to data
        roster = self.data['Nickname'].apply(getValue)
        roster.name = roster_name
        
        # append to data
        self.data = pd.concat([self.data, roster], axis = 'columns')
                              
        return pulp.LpStatus[problem.status]
        
    
    def _strategy_single(self, point_col):
        """
        Find the global LP optimal solution and append to self.data. Results
        in a single roster.
        
        Appends to self.data.
        
        Returns: number of optimal solutions,
                 number of rosters
        """
        result = self._solveProblem(point_col)
        
        return 1 if result == 'Optimal' else 0, 1
    
    
    def _strategy_iterative(self, point_col, visual=True):
        """
        Finds a neighborhood of optimal solutions by dropping players in
        optimal roster, secondary, and tertiary rosters for a total of
        three iterations. 
        
        Appends to self.data.

        Returns: number of optimal solutions,
                 number of rosters        
        """
        start = datetime.now()
        total_optimal = 0
        total_rosters = 0
        
        # initial problem
        result = self._solveProblem(point_col)
        init_sol_players = list(self.data[self.data['Roster0']==1]['Nickname'])
        
        if result != 'Optimal':
            raise ValueError("Strategy Iterative's initial solution is not optimal!")

        total_optimal += 1
        total_rosters += 1
        
        # use initial solution to create powerset and iterate
        pset = self._powerset(init_sol_players)
        
        counter = 1
        # initial iteration
        optimal = 0
        rosters = 0
        for s in pset:
            result = self._solveProblem(point_col, f'Roster{counter}', s)
            if result == 'Optimal':
                optimal += 1
            rosters += 1
            counter += 1
            
        total_optimal += optimal
        total_rosters += rosters
        
        if visual:
            stop = datetime.now()
            print(f'\nInitial iteration completed in {stop - start}.')
            print(f'Optimal solutions: {optimal} / {rosters}')
            
        # second iteration
        start_roster = self.data.columns[-1]
        print('\nStarting second iteration...')
        
        secondary_solution = list(self.data[self.data[start_roster]==1]['Nickname'])
        pset = self._powerset(secondary_solution)
        
        optimal = 0
        rosters = 0
        for s in pset:
            # drop initial solution and subset
            result = self._solveProblem(point_col, f'Roster{counter}',
                                        init_sol_players + list(s))
            if result == 'Optimal':
                optimal += 1
            rosters += 1
            counter += 1
        
        total_optimal += optimal
        total_rosters += rosters
        
        if visual:
            print(f'\nSecond iteration completed in {datetime.now() - stop}.')
            stop = datetime.now()
            print(f'Optimal solutions: {optimal} / {rosters}')
            
        # third iteration
        start_roster = self.data.columns[-1]
        print('\nStarting third iteration...')
        
        tertiary_solution = list(self.data[self.data[start_roster]==1]['Nickname'])
        pset = self._powerset(tertiary_solution)
        
        optimal = 0
        rosters = 0
        for s in pset:
            # drop initial solution, secondary solution, and subset
            result = self._solveProblem(point_col, f'Roster{counter}',
                                        init_sol_players + secondary_solution + list(s))
            if result == 'Optimal':
                optimal += 1
            rosters += 1
            counter += 1
        
        total_optimal += optimal
        total_rosters += rosters
        
        if visual:
            print(f'\nThird iteration completed in {datetime.now() - stop}.')
            stop = datetime.now()
            print(f'Optimal solutions: {optimal} / {rosters}')
        
        return total_optimal, total_rosters
    
    
    def _strategy_reimagined(self, point_col, visual=True, max_te=2):
        """
        Finds a neighborhood of optimal solutions by dropping players in optimal
        roster and then optimal plus one optimal player at a time for a total
        of nine iterations.
        
        Appends to self.data.
        
        Returns: number of optimal solutions,
                 number of rosters
        """
        start = datetime.now()
        total_optimal = 0
        total_rosters = 0
        
        # initial problem
        result = self._solveProblem(point_col, max_te=max_te)
        init_sol_players = list(self.data[self.data['Roster0']==1]['Nickname'])
        
        if result != 'Optimal':
            raise ValueError("Strategy Reimagined's initial solution is not optimal!")
            
        total_optimal += 1
        total_rosters += 1
        
        # use initial solution to create powerset and iterate
        pset = self._powerset(init_sol_players)
        
        counter = 1
        # intial iteration
        optimal = 0
        rosters = 0
        for s in pset:
            result = self._solveProblem(point_col, f'Roster{counter}', s, max_te=max_te)
            if result == 'Optimal':
                optimal += 1
            rosters += 1
            counter +=1
        
        total_optimal += optimal
        total_rosters += rosters
        
        if visual:
            stop = datetime.now()
            print(f'\nInitial iteration completed in {stop - start}.')
            print(f'Optimal solutions: {optimal} / {rosters}')
        
        # alternate iterations
        for j in range(1, 10):
            start_roster = f'Roster{j}'
            already_dropped = pset[j-1][0]
            alt_sol_players = list(self.data[self.data[start_roster]==1]['Nickname'])
            alt_pset = self._powerset(alt_sol_players)
            
            if visual:
                print(f'\nDropped {already_dropped}. Using {start_roster} as basis...')
            
            optimal = 0
            rosters = 0
            for s in alt_pset:
                result = self._solveProblem(point_col, f'Roster{counter}',
                                            tuple(already_dropped) + s,
                                            max_te=max_te)
                if result == 'Optimal':
                    optimal += 1
                rosters += 1
                counter += 1
            
            total_optimal += optimal
            total_rosters += rosters
            
            if visual:
                print(f'Iteration {j} completed in {datetime.now() - stop}.')
                stop = datetime.now() # reset stop
                print(f'Optimal solutions: {optimal} / {rosters}')
        
        return total_optimal, total_rosters
    
    
    def _summarize_rosters(self):
        """
        Assumes rosters have been generated. Updates self.rosters with summary stats.
        """
        roster_cols = self.data.columns[self.data.columns.str.contains('Roster')]
        
        for rost in roster_cols:
            dat_select = self.data[self.data[rost] == 1]
            
            # summary of selected
            sal = dat_select['Salary'].sum()
            proj = dat_select['Projected'].sum()
            avg = dat_select['Hist_Avg'].sum()
            std = (dat_select['Hist_Std'] ** 2).sum() ** .5
            per = dat_select['Proj_Percent'].sum()
            
            # k is distance from original (number of dropped players)
            k = self.data[self.data[rost] == -1].shape[0]
            
            self.rosters.loc[rost] = [
                sal, round(proj, 1), round(avg, 2), round(std, 2), per, k]
        
        return True
    
    
    def generateRosters(self, strategy, salary=None, min_pts=None, point_col='Projected',
                        drop_injuries=[], drop_players_initial=[],
                        visual = True, max_te = 2):
        """
        Generates rosters by using the given strategy.
        
        Modifies
        --------
        self.data (clearRosters and then append)

        Parameters
        ----------
        strategy : (single, iterative, reimagined)
        salary : maximum salary
        min_pts : pre-processing filter
        drop_injuries : pre-processing filter
        visual : prints status updates during generation
        
        """
        if strategy not in ['single', 'iterative', 'reimagined']:
            raise ValueError(f'Strategy "{strategy}" was passed. Must be one of "single", "iterative", "reimagined".')
        
        start = datetime.now()
        
        self._strategy = strategy
        # initialize parameters
        if self._sport == 'NFL':
            self._salary_limit = 60000
            if salary:
                self._salary_limit = salary
            self._min_points = 3
            if min_pts:
                self._min_points = min_pts
            self._drop_injuries = ['Q', 'D', 'O', 'NA', 'IR']
            if drop_injuries:
                self._drop_injuries = drop_injuries
            self._drop_players = []
            if drop_players_initial:
                self._drop_players = drop_players_initial
                
        if self._sport == 'NBA':
            # TODO implement NBA
            raise NotImplementedError()
         
        # clear rosters
        self.clearRosters()
        
        # prefiltering
        dat = self.data
        
        # keep all defenses
        # TODO this is NFL specific, clean up for any sport
        k_def = (dat['DST'] == 1) & (~dat['Nickname'].isin(self._drop_players))
        
        # keep players with at least min_points
        k_points = dat[point_col] >= self._min_points
        
        # keep players without specified injuries
        k_injuries = ~dat['Injury Indicator'].isin(self._drop_injuries)
        
        # keep players not purposefully dropped
        k_players = ~dat['Nickname'].isin(self._drop_players)
        
        # keep players with enough historical information
        k_std = ~dat['Hist_Std'].isna()
        
        # master mask
        keep = k_def | (k_points & k_injuries & k_players & k_std)
        self.tracking['preprocessing_drops'] = dat[~keep]
        self.data = dat[keep]
        
        stop_preprocessing = datetime.now()
        print(f'Preprocessing completed in {stop_preprocessing - start}.')
        
        # strategies
        print(f'\nBeginning strategy: {strategy}')
        print('--------------------------------------------------')
        
        if strategy == 'single':
            results = self._strategy_single(point_col)
                        
        if strategy == 'iterative':
            results = self._strategy_iterative(point_col, visual)
            
        if strategy == 'reimagined':
            results = self._strategy_reimagined(point_col, visual, max_te=max_te)
        
        stop_strategy = datetime.now()
        
        # summarize rosters
        print('--------------------------------------------------')
        print(f'\nStrategy completed in {stop_strategy - stop_preprocessing}.')
        print(f'Optimal results: {results[0]} / {results[1]}')
        
        print('\nSummarizing rosters...')
        self._summarize_rosters()
        
        stop_summarize = datetime.now()
        print(f'Rosters summarized in {stop_summarize - stop_strategy}.\n')
        
        print(f'Total time: {stop_summarize - start}.')
        
        # return total time and number of non-optimal solutions
        return results
            
        
    def selectRosters(self, q_points, q_std=1.00):
        """
        Select roster from generated rosters.
        """
        # check if q_std is None
        if not q_std:
            q_std = 1.00
        self._q_points = q_points
        self._q_std = q_std
        
        top = self.rosters[
            (self.rosters['Projected'] >= self.rosters['Projected'].quantile(self._q_points)) &
            (self.rosters['Hist_Std'] <= self.rosters['Hist_Std'].quantile(self._q_std))]
        
        # count appearances
        self.data['Top Appearances'] = (
            self.data.loc[:, self.data.columns.isin(top.index)] > 0
            ).sum(axis='columns')
    
        # score top rosters
        scores = top.index.to_series().apply(
            lambda x: self.data[self.data[x]==1]['Top Appearances'].sum())
        scores.name = 'Score'
        top = pd.concat([top, scores], axis='columns')
        
        top_sorted = top.sort_values(['Score', 'K'], ascending=[False, True])
        self._selection = top_sorted.iloc[0].name
        
        # append columns to self.rosters
        # top roster flag
        self.rosters['Top'] = self.rosters.index.isin(top.index)
        
        # score each roster with top appearances
        self.rosters['Top Appearances'] = self.rosters.index.to_series().apply(
            lambda x: self.data[self.data[x]==1]['Top Appearances'].sum())
        
        return top_sorted
    
    
    # Rebuild Rosters, Rename Data, Export Game -------------------------------
    # TODO clean up
    
    def rebuild_data(self):
        
        dat = self.data
        
        # rename columns
        col_name_map = {
            'Injury' : 'Injury Indicator',
            'FPPG' : 'Projected',
            'MrPeriod' : 'MR_Period',
            'ROST' : 'Proj_Percent',
            'AvgActual' : 'Hist_Avg',
            'StdActual' : 'Hist_Std',
            'Actual_AVG' : 'Hist_Avg',
            'Actual_STD' : 'Hist_Std'
        }
        dat.columns = list(dat.columns.to_series().apply(lambda x: col_name_map.get(x, x)))
        
        # get historicals
        self.historical = self.loadFantasyProsActuals()[0]
        hist_sum = self.summarizeActuals()
        
        if 'Proj_Percent' not in dat.columns:
            dat = dat.merge(
                hist_sum['Proj_Percent'], how = 'left', left_on = 'Nickname', right_index=True)
        
        if 'N_Games' not in dat.columns:
            dat = dat.merge(
                hist_sum['N_Games'], how = 'left', left_on = 'Nickname', right_index=True)
        
        # organize columns and save 
        attr_cols = ['Nickname', 'Position', 'Salary', 'Injury Indicator',
                      'QB', 'RB', 'WR', 'TE', 'DST', 'FLEX', 'Projected',
                      'MR_Period', 'N_Games', 'Hist_Avg', 'Hist_Std', 'Proj_Percent']
        rost_cols = list(dat.columns[dat.columns.str.contains('Roster')])
        self.data = dat[attr_cols + rost_cols]
        
        return True
    
    
    def rebuild_rosters(self):
        
        # initialize
        self.rosters = pd.DataFrame(
            columns = ['Salary', 'Projected', 'Hist_Avg', 'Hist_Std', 'Proj_Percent', 'K'])
        
        self._summarize_rosters()
        
        return True
        
    
    def rebuild_game(self):
        
        self.rebuild_data()
        self.rebuild_rosters()
        self.selectRosters(self._q_points, self._q_std)

        
    # PLOTTING FUNCTIONS ------------------------------------------------------
    
    def generateEfficientFrontier(self):
        # TODO
        pass
    
    
    def graphEfficientFrontier(self, with_rosters=False):
        # TODO
        pass
    
    
    def graphPlayersByPosition(self):
        # TODO
        pass
    
    
    def graphRosters(self, points='projected', how='std'):
        # TODO
        pass