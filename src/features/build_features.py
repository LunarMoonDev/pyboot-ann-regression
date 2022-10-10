# import libraries
from cgi import test
from ctypes import sizeof
import pandas as pd
import numpy as np
import torch


# constants
TAXI_CSV = './data/raw/NYCTaxiFares.csv'

# util methods
def haversine_distance(df, lat1, long1, lat2, long2):
    '''
    Calculates the distance between 2 sets of GPS coordinates in df
    '''
    r = 6371 # average radius of earth in km

    ph1 = np.radians(df[lat1])
    ph2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[long2] - df[long1])

    a = np.sin(delta_phi/2) ** 2 + np.cos(ph1) * np.cos(ph2) * np.sin(delta_lambda/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r *c)

    return d

# data prep
df = pd.read_csv(TAXI_CSV)
print("content of main dataframe (head): \n", df.head())

df['dist_km'] = haversine_distance(df, 
                                    'pickup_latitude', 
                                    'pickup_longitude',
                                    'dropoff_latitude',
                                    'dropoff_longitude')
print("content of main dataframe (head): \n", df.head())

df['EDTdate'] = pd.to_datetime(df['pickup_datetime'].str[:19]) - pd.Timedelta(hours = 4)
df['Hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where(df['Hour'] < 12, 'am', 'pm')
df['Weekday'] = df['EDTdate'].dt.strftime("%a")
print("content of main dataframe (head): \n", df.head())

# [dev]: separting categories from continious
cat_cols = ['Hour', 'AMorPM', 'Weekday']
cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_longitude', 'dropoff_latitude', 
             'passenger_count', 'dist_km']
y_col = ['fare_amount']

for cat in cat_cols:
    df[cat] = df[cat].astype('category')
print("types of main dataframe's columns: \n", df.dtypes)

# [dev]: making a cat code stack
hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values

cats = np.stack([hr, ampm, wkdy], 1)
print("contents of the categorical code stack: \n", cats[:5])
cats = torch.tensor(cats, dtype=torch.int64)

conts = np.stack([df[col].values for col in cont_cols], 1)
print("contents of the continious code stack: \n", conts[:5])
conts = torch.tensor(conts, dtype=torch.float)

y = torch.tensor(df[y_col].values, dtype=torch.float).reshape(-1, 1)
print("contents of y value: \n", y[:5])

print("\n sizes of inputs: \n")
print(cats.shape)
print(conts.shape)
print(y.shape)

# print("cats:", type(cats))
# print("conts:", type(conts))
# print("y:", type(y))

pd.DataFrame(cats.numpy()).to_csv('./data/interim/cats.csv', header=None, index=False)
pd.DataFrame(conts.numpy()).to_csv('./data/interim/conts.csv', header=None, index=False)
pd.DataFrame(y.numpy()).to_csv('./data/interim/y.csv')

# not the actual batchsize but this limits the data we are using as input
batch_size = 60000 
test_size = int(batch_size * 0.2)

# building the test and train data
cat_train = cats[: batch_size - test_size]
cat_test = cats[batch_size - test_size: batch_size]
con_train = conts[: batch_size - test_size]
con_test = conts[batch_size - test_size: batch_size]
y_train = y[: batch_size - test_size]
y_test = y[batch_size - test_size: batch_size]

pd.DataFrame(cat_train.numpy()).to_csv('./data/processed/cat_train.csv', header=None, index=False)
pd.DataFrame(cat_test.numpy()).to_csv('./data/processed/cat_test.csv', header=None, index=False)
pd.DataFrame(con_train.numpy()).to_csv('./data/processed/con_train.csv', header=None, index=False)
pd.DataFrame(con_test.numpy()).to_csv('./data/processed/con_test.csv', header=None, index=False)
pd.DataFrame(y_train.numpy()).to_csv('./data/processed/y_train.csv', header=None, index=False)
pd.DataFrame(y_test.numpy()).to_csv('./data/processed/y_test.csv', header=None, index=False)
