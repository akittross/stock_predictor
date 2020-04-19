
import pandas as pd

idf = pd.read_csv("djia_close.csv")
odf = pd.DataFrame(columns=['date', 'change'])
prev = idf['close'][0]
first = True
for _, row in idf.iterrows():
    if not first:
        current_date = row['date']
        change = row['close'] - prev
        change = change / prev
        new_row = pd.DataFrame(data={'date' : [current_date], 'change' : [change]})
        odf = odf.append(new_row)
        prev = row['close']
    else:
        first = False

odf.to_csv('djia_change.csv', index=False)