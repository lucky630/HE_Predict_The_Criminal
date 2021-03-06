import numpy as np
import pandas as pd
import sklearn.preprocessing
from scipy.special import erfinv

def hot_encoder(df, columns):
	one_hot = {c: list(df[c].unique()) for c in columns}
	for c in one_hot:
		for val in one_hot[c]:
			df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
	return df

def rank_gauss(x):
    # x is numpy vector
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

print('Reading data')

#df_train = pd.read_csv('train.csv')
#df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('criminal_train_clean.csv')
df_test = pd.read_csv('criminal_test.csv')

train_target = df_train['Criminal'].values

ntrain = df_train.shape[0]
ntest  = df_test.shape[0]

print('Train data with {} rows'.format(ntrain))
print('Test data with {} rows'.format(ntest))

print('Transforming data')

feature_cols = [c for c in df_train.columns if c not in ['PERID','Criminal']]
#keep_cols    = [c for c in feature_cols]
keep_cols    = [c for c in feature_cols if not c.startswith('ANALWT_C')] 
#cat_cols     = [c for c in keep_cols if '_cat' in c]

df_all = pd.concat([df_train[keep_cols], df_test[keep_cols]])

df_all = hot_encoder(df_all, keep_cols)

data_all = df_all.values

cols = data_all.shape[1]

print(df_all.columns)

assert len(df_all.columns) == cols

print('Final data with {} columns'.format(cols))

for i in range(cols):
	u = np.unique(data_all[:,i])
	if u.shape[0] > 3:
		data_all[:,i] = rank_gauss(data_all[:,i])

train_data = data_all[0:ntrain,:]
test_data  = data_all[ntrain:,:]

assert train_data.shape[0] == ntrain
assert test_data.shape[0]  == ntest

print('Saving data')

np.save('train_target_org.npy', train_target)
np.save('train_data_org.npy', train_data)
np.save('test_data_org.npy', test_data)
