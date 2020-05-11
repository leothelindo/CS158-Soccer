try :
    from tqdm import tqdm
    #raise
except :
    def tqdm(items, **kwargs) :
        if 'desc' in kwargs:
            print(kwargs['desc'])
        for it in items :
            yield it

import xlsxwriter 
import pandas as pd
import os

# project-specific helper libraries
import soccer_config
import soccer_mapping

def setup_ids(path):
    """
    Saves csv of shape (n_samples, 2)
        Labels.
        columns:
            MatchID (int)
            Outcome (int, -1 for Loss, 0 for Tie, +1 for Win)
    """

    # Make new xlsx file
    lab_path = os.path.join(path, 'labels.xlsx')
    wb = xlsxwriter.Workbook(lab_path) 
    
    # Make new worksheet in xlsx file 
    ws = wb.add_worksheet() 
    ws.write('A1', 'MatchID') 
    ws.write('B1', 'Outcome') 

    # Read in match data to determine match winner
    df = pd.read_csv(os.path.join(path, 'data.csv'))
    
    match_id = 1
    row = 1
    for index, r in df.iterrows():
        # Use the worksheet object to write data 
        ws.write(row, 0, match_id) 
        
        # Determine match outcome with respect to home team
        if r['FTHG'] > r['FTAG']:
            ws.write(row, 1, 1)
        elif r['FTHG'] < r['FTAG']:
            ws.write(row, 1, -1)
        else:
            ws.write(row, 1, 0)
        row += 1
        match_id += 1

    # Close the Excel file via the close() method. 
    wb.close()

    # Convert to csv
    df = pd.read_excel(lab_path)
    df.to_csv(os.path.join(path, 'labels.csv'), sep=',', index=False)



def get_raw_data(path, n=None):
    """
    Adds MatchID mapping to each game.
    Read raw data from <path>/labels.csv and <data>/files/*.csv, keeping only the first n examples.
    
    Parameters
    --------------------
    path : string
        Data directory.
    
    n : int
        Number of examples to retain.
    """

    # Read labels and keep first n
    df_labels = pd.read_csv(os.path.join(path, 'labels.csv'))
    if n is not None :
        df_labels = df_labels[:n]
    ids = df_labels['MatchID']

    df = pd.read_csv(os.path.join(path, 'data.csv'))

    data = df.copy(deep=True)
    data.insert(0, "MatchID", ids)
    data.to_csv(os.path.join(soccer_config.PROCESSED_DATA_DIR, 'data' + str(n) + '.csv'), index=False)

def remap_teams(path):
    df = pd.read_csv(os.path.join(path, 'data3733.csv'))
    data = df.copy(deep=True)

    numeric = soccer_mapping.transform()
    data['HomeTeam'] = data['HomeTeam'].map(numeric)
    data['AwayTeam'] = data['AwayTeam'].map(numeric)
    data = data.drop(['Referee', 'FTHG', 'FTAG', 'FTR', 'HTR'], axis=1)
    data.to_csv(os.path.join(soccer_config.PROCESSED_DATA_DIR, 'data3733_num.csv'), index=False)
    

def main():
    DATA_DIR = '..\data'
    raw = soccer_config.RAW_DATA_DIR
    processed = soccer_config.PROCESSED_DATA_DIR
    NRECORDS = 3733     # number of match records

    # setup_ids(raw)
    # get_raw_data(raw, 3733)
    remap_teams(processed)

if __name__ == '__main__':
    main()
    