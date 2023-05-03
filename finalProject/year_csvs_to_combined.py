# This file takes the csvs from data folder and combines them into one csv by cleaning, renaming columns etc

import pandas as  pd
import os

# read in the csv files
dir = "data"

dfs = []
columns = []
for filename in os.listdir(dir):
    print(f'Rewriting {filename}')
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(dir,filename))
        # remove the 'Respondent Frequency' column if it exists
        if 'Respondent Frequency' in df.columns:
            df = df.drop(columns=['Respondent Frequency'])
        # remove the 'Balancing Authority Code' column if it exists
        if 'Balancing Authority Code' in df.columns:
            df = df.drop(columns=['Balancing Authority Code'])
        if 'BA_CODE' in df.columns:
            df = df.drop(columns=['BA_CODE'])
        # Rename the Id columns to ID if it exists
        if 'Plant ID' in df.columns:
            df = df.rename(columns={'Plant ID':'Plant Id',
                                    'Generator ID':'Generator Id',
                                    'Operator ID':'Operator Id',
                                    'State':'Plant State',
                                    'EIA Sector Number':'Sector Number',
                                    'Prime Mover Type':'Reported Prime Mover'})
        if 'EIA Sector Number' in df.columns:
            df = df.rename(columns={'EIA Sector Number':'Sector Number'})
        if 'Net Generation Year to Date' in df.columns:
            df = df.rename(columns={'Net Generation Year to Date':'Net Generation Year To Date'})
        if 'Prime Mover Type' in df.columns:
            df = df.rename(columns={'Prime Mover Type':'Reported Prime Mover'})
        if 'State' in df.columns:
            df = df.rename(columns={'State':'Plant State'})
        if 'Combined Heat And  Power Plant' in df.columns:
            df = df.rename(columns={'Combined Heat And  Power Plant':'Combined Heat And Power Plant'})
        if 'Combined Heat & Power Plant' in df.columns:
            df = df.rename(columns={'Combined Heat & Power Plant':'Combined Heat And Power Plant'})

        # resave df to csv
        df.to_csv('data/'+filename,index=None)

# read in the csv files
dir = "data"

dfs = []
columns = []
month_dict = {
    'Net Generation January':1,
    'Net Generation February':2,
    'Net Generation March':3,
    'Net Generation April':4,
    'Net Generation May':5,
    'Net Generation June':6,
    'Net Generation July':7,
    'Net Generation August':8,
    'Net Generation September':9,
    'Net Generation October':10,
    'Net Generation November':11,
    'Net Generation December':12
}
for filename in os.listdir(dir):
    if filename.endswith('.csv'):
        filePath = os.path.join(dir,filename)
        df = pd.read_csv(filePath)
        # For each net generation column, change the column name to the month and year
        for col in df.columns:
            if 'Net Generation' in col and col in month_dict:
                df = df.rename(columns={col:f'{month_dict[col]}-{df["YEAR"][0]}'})
        # Concatenate the Generator Id and Plant Id columns
        df['Generator Id'] = df['Plant Id'].astype(str) + '-' + df['Generator Id'].astype(str) + '-' + df['Sector Number'].astype(str)
        # Use the Generator Id as the index
        df = df.set_index('Generator Id')
        # Drop the Plant Id column
        df = df.drop(columns=['Plant Id', 'Combined Heat And Power Plant', 'Plant Name', 'Operator Name',
        'Plant State', 'Sector Number', 'Reported Prime Mover', 'Plant State', 'YEAR', 'NAICS Code',
        'Operator Id', 'Census Region', 'NERC Region', 'Sector Name', 'Net Generation Year To Date'])

        # Drop rows that have duplicate Generator Ids
        df = df[~df.index.duplicated(keep='first')]


        print(filename, df.shape)
        print(df.columns)
        dfs.append(df)

# join the dataframes by index
df = pd.concat(dfs, axis=1, join='outer')
# Transpose the dataframe so the the colunmns are the Generator Ids
df = df.transpose()
# Convert the index to a datetime
df.index = pd.to_datetime(df.index)
# Sort the rows by the index
df = df.sort_index()
# Save the dataframe to a csv
df.to_csv('combined.csv')