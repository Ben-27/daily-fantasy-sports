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
            self.rosters = pd.DataFrame(columns=['Salary', 'Projected', 'STD', 'K'])
            
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
            self.data = self.data.merge(
                self.actuals[['Nickname', 'FPTS']],
                how = 'left', left_on = 'Nickname', right_on='Nickname')
            # merge actual roster points to rosters dataframe
            rosters = self.data[
                self.data.columns[list(self.data.columns).index('Roster0') : -1]
                ].replace(-1, 0)
            points = self.data[self.data.columns[-1]]
            r_points = rosters.multiply(points.fillna(0), axis='rows').sum(axis='rows')
            self.rosters['Actual'] = r_points
            
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
            temp = pd.read_excel(folder + self._historical, sheet_name=pos)
            dat.append(temp[['Player', 'FPTS', 'Season', 'WeekNum', 'G']])
            
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
            ['Nickname', 'Period', 'Season', 'WeekNum', 'G', 'FPTS', 'FPTS']]
        
        # grab n most recent games
        top = act.groupby('Nickname').head(self._n_historical)
        
        # prep columns, aggregate, and clean up
        top.columns = ['Nickname', 'MR_Period', 'MR_Season', 'MR_WeekNum',
                       'N_Games', 'Actual_AVG', 'Actual_STD']
        agg = top.groupby('Nickname').aggregate(
            {'MR_Period':'first', 'MR_Season':'first', 'MR_WeekNum':'first',
             'N_Games':'sum', 'Actual_AVG':'mean', 'Actual_STD':'std'})
        agg['Actual_AVG'] = round(agg['Actual_AVG'], 2)
        agg['Actual_STD'] = round(agg['Actual_STD'], 2)
        return agg
    
    
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
        self.rosters = pd.read_excel(full_file_name + '.xlsx',
                                     sheet_name = 'rosters',
                                     index_col = 0)
        return True
    
    
    def export_game(self, file_name=None):
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
        if os.path.exists(save_folder + file_name) or \
            os.path.exists(save_folder + file_name + '.xlsx'):
            print(f'"{save_folder}{file_name}" already exists!')
            return False
        else:
            # truncate actuals
            data = self.data[self.data.columns[:-1]]
            rosters = self.rosters[self.rosters.columns[:-1]]
            # save meta, data, and rosters to Excel
            with pd.ExcelWriter(save_folder + file_name + '.xlsx') as writer:
                meta.to_excel(writer, sheet_name = 'meta')
                data.to_excel(writer, sheet_name = 'data')
                rosters.to_excel(writer, sheet_name = 'rosters')
            # save data as pickle file
            data.to_pickle(save_folder + file_name)
        return True
    
    
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
        self.rosters = pd.DataFrame(columns=['Salary', 'Projected', 'STD', 'K'])
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
    
    
    def _defineProblem(self, dat, point_col='FPPG', drop_players=[],
                       variance_type=None, variance_target=None):
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
                problem += pulp.lpSum(players * A.iloc[3]) <= 2, 'TE Max Contraint'
                problem += pulp.lpSum(players * A.iloc[4]) == 1, 'DST Contraint'
                problem += pulp.lpSum(players * A.iloc[5]) == 7, 'FLEX Contraint'
                problem += pulp.lpSum(players * A.iloc[6]) <= self._salary_limit, 'Salary Contraint'
                return problem
            
            # TODO implement NBA
            if self._sport == 'NBA':
                raise NotImplementedError('_defineProblem for NBA not yet implemented.')
            
            return False
       
        
    def _solveProblem(self, point_col, roster_name='Roster0', drop_players=[]):
        """
        Creates and solves LpProblem from data and appends the roster to the dataframe.
    
        Modifies: self.data
        
        Returns: solution status
        """
        problem = self._defineProblem(self.data, point_col, drop_players)
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

    # todo destroy this shit
    def test_strategy_iterative(self, point_col, visual=True):
        """
        Finds a neighborhood of optimal solutions by dropping players in
        optimal roster, secondary, and tertiary rosters for a total of
        three iterations. 
        
        Appends to self.data.

        Returns: number of optimal solutions,
                 number of rosters        
        """
        ROSTERS = []
        start = datetime.now()
        total_optimal = 0
        total_rosters = 0
        
        # initial problem
        result = self._solveProblem(point_col)
        init_sol_players = list(self.data[self.data['Roster0']==1]['Nickname'])
        if result[0] != 'Optimal':
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
            result = self.test_solveProblem(point_col, f'Roster{counter}', s)
            if result[0] == 'Optimal':
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
            result = self.test_solveProblem(point_col, f'Roster{counter}',
                                        init_sol_players + list(s))
            if result[0] == 'Optimal':
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
            result = self.test_solveProblem(point_col, f'Roster{counter}',
                                        init_sol_players + secondary_solution + list(s))
            if result[0] == 'Optimal':
                optimal += 1
            rosters += 1
            counter += 1
        
        total_optimal += optimal
        total_rosters += rosters
        
        if visual:
            print(f'\nThird iteration completed in {datetime.now() - stop}.')
            stop = datetime.now()
            print(f'Optimal solutions: {optimal} / {rosters}')
            
        self.data = pd.concat([self.data] + ROSTERS, axis='columns')
        return total_optimal, total_rosters
        
    
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
    
    
    def _strategy_reimagined(self, point_col, visual=True):
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
        result = self._solveProblem(point_col)
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
            result = self._solveProblem(point_col, f'Roster{counter}', s)
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
                                            tuple(already_dropped) + s)
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
            std = (dat_select['Actual_STD'] ** 2).sum() ** .5
            
            # k is distance from original (number of dropped players)
            k = self.data[self.data[rost] == -1].shape[0]
            
            self.rosters.loc[rost] = [sal, round(proj, 1), round(std, 2), k]
        
        return True
    
    def generateRosters(self, strategy, salary=None, min_pts=None, point_col='Projected',
                        drop_injuries=[], drop_players_initial=[],
                        visual = True):
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
        k_std = ~dat['Actual_STD'].isna()
        
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
            results = self._strategy_reimagined(point_col, visual)
        
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
        self._q_points = q_points
        self._q_std = q_std
        
        top = self.rosters[
            (self.rosters['Projected'] >= self.rosters['Projected'].quantile(self._q_points)) &
            (self.rosters['STD'] <= self.rosters['STD'].quantile(self._q_std))]
        
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
        
        return top_sorted
        
        
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