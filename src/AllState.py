
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Print all rows and columns. Dont hide any
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# Use this seed for all random states
seed = 0

# for data preprocessing using normalization
scaler_x = StandardScaler()
scaler_y = StandardScaler()


# In[2]:


# XGBoost bug hot-fix:
# XGBoost cannot predict large test dataset at one go, so we divide test set into small chuck

def getPred(model,X_val):
    chunk = 5000  #chunk row size
    X_val_chunks = [X_val[i:i+chunk] for i in range(0,X_val.shape[0],chunk)]

    pred = []
    for X_val_chunk in X_val_chunks:
        pred.append(model.predict(X_val_chunk))
    pred = np.concatenate(pred)
    return pred


# ## Load raw data:
# - In this step, we save `'ID'` column for test set so we can construct `submission.csv` file after prediction.

# In[3]:


raw_train = pd.read_csv("../data/train.csv")
raw_test = pd.read_csv("../data/test.csv")

# Save the id's for submission file
ID = raw_test['id']

# drop 'id' column
raw_train.drop('id',axis=1, inplace=True)
raw_test.drop('id',axis=1, inplace=True)

#Display the first row to get a feel of the data
print(raw_train.head(1))


# ## Split dataset:
# - index 0 - 115 are category
# - index 116(included) onwards are numerical

# In[4]:


#scaler = StandardScaler().fit(raw_train)
last_discrete_idx = 116

raw_train_discrete = raw_train.iloc[:,:last_discrete_idx]
raw_train_continuous = raw_train.iloc[:,last_discrete_idx:-1]
raw_trainY = raw_train[['loss']]

raw_test_discrete = raw_test.iloc[:,:last_discrete_idx]
raw_test_continuous = raw_test.iloc[:,last_discrete_idx:]


# ## Data statistics:
# - Shape
# - Description
# - Skew

# In[5]:


temp = pd.concat([raw_train_discrete, raw_train_continuous, raw_trainY], axis=1)

print(temp.shape)
# Observe that means are 0s and standard deviations are 1s
print(temp.describe())
print(temp.skew())


# ## Data Transformation:
# - Skew correction
# - We use `log shift` method to improve the skewness of the `'loss'` column
# - We try shift values [0, 1, 10, 100, 500, 1000] and plot the graph of it.
# 
# **Result: **
# 
# - Best shift is 0
# 
# **Take Note: **
# 
# - We have to use `np.exp()` later to convert back the `'loss'` after prediction

# In[6]:


temp = raw_trainY['loss']
original_skew = temp.skew()
print('Skewness without log shift: ' + str(original_skew))

shifts = [0, 1, 10, 100, 500, 1000]
temp_result = []

for shift in shifts:
    shifted = np.log(temp + shift)
    temp_result.append(shifted.skew())

val, idx = min((val, idx) for (idx, val) in enumerate(temp_result))
best_shift = shifts[idx]

print('Best shift: ' + str(shifts[idx]))
print('Skewness with log shift: ' + str(val))

plt.plot(shifts, temp_result)
plt.show()

raw_trainY['loss'] = np.log(raw_trainY['loss'] + best_shift)


# ## Data Pre Processing:
# - Normalization (Z-Scoring)
# 
# **Take Note: **
# 
# - We split data to X (all the features) and Y (loss) and perform normalization separately. This is so that we can use `scaler_y` to inverse transform the prediction of loss later.
# - We only use the train set to fit the normalization.

# In[7]:


scaler_x.fit(raw_train_continuous)
scaler_y.fit(raw_trainY)

# Save columns name
col_name_X = raw_train_continuous.columns.values
col_name_Y = raw_trainY.columns.values

# transform
clean_train_continuous = scaler_x.transform(raw_train_continuous)
clean_trainY = scaler_y.transform(raw_trainY)
clean_test_continuous = scaler_x.transform(raw_test_continuous)

clean_train_continuous = pd.DataFrame(data=clean_train_continuous, columns=col_name_X)
clean_trainY = pd.DataFrame(data=clean_trainY, columns=col_name_Y)
clean_test_continuous = pd.DataFrame(data=clean_test_continuous, columns=col_name_X)


# ## Data Visualization:
# - Categorical attributes
# - It can be observed that cat1 to 98 have significantly less number of categories than cat99 to 116.

# In[8]:


# Count of each label in each category
try:
    count_result = pd.read_pickle('../intermediate/count_result')
except FileNotFoundError:
    temp = pd.concat([raw_train_discrete, raw_test_discrete])
    count_result = temp.apply(pd.value_counts)
    count_result.to_pickle('../intermediate/count_result')

#names of all the columns
cols = count_result.columns

# Plot count plot for all attributes in a 29x4 grid
n_cols = 4
n_rows = 29
fig, axes = plt.subplots(n_rows, n_cols, sharey=True, figsize=(12, 100))
for i in range(n_rows):
    for j in range(n_cols):
        col_name = cols[i*n_cols+j]
        temp = count_result[col_name]
        temp = temp.dropna()
        axes[i, j].hist(temp.index.values.tolist(), weights=temp.tolist())
        axes[i, j].set_title(col_name)
plt.savefig('../intermediate/count_plot.png', dpi=100)


# ## Feature Engineering:
# ### Motivation:
# - Using One-Hot encoding to all categorical data may increase the number of features substantially and this requires long computational time
# ### Approach:
# - For features with small number of categories (cat1-98), we use one-hot encoding
# - For features with large number of categories (cat99-116), we use ordinal encoding

# In[9]:


n_train = raw_train_discrete.shape[0]
n_test = raw_test_discrete.shape[0]

split = 98

one_hot_train = raw_train_discrete.iloc[:,:split]
one_hot_test = raw_test_discrete.iloc[:,:split]
one_hot_temp = pd.concat([one_hot_train, one_hot_test])

ordinal_train = raw_train_discrete.iloc[:,split:]
ordinal_test = raw_test_discrete.iloc[:,split:]
ordinal_temp = pd.concat([ordinal_train, ordinal_test])

# One-Hot encoding
one_hot_temp = pd.get_dummies(one_hot_temp)
# Ordinal encoding
from sklearn.preprocessing import LabelEncoder
ordinal_temp = ordinal_temp.apply(LabelEncoder().fit_transform)

encoded = pd.concat([one_hot_temp, ordinal_temp], axis=1)

print(encoded.shape)

raw_train_discrete_encoded = encoded.iloc[:n_train,:]
raw_test_discrete_encoded = encoded.iloc[n_train:,:]


# ## Data Preparation:
# - Split into train and validation
# - We use K-Fold method with k = 5
# - We also declare `mean_absolute_error` as a scoring parameter

# In[10]:


XY_train = pd.concat([raw_train_discrete_encoded, clean_train_continuous, clean_trainY], axis=1)
X_test = pd.concat([raw_test_discrete_encoded, clean_test_continuous], axis=1)

print('Number of dataset: ')
print('Train: ' + str(XY_train.shape[0]))
print('Test: ' + str(X_test.shape[0]))

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=seed, shuffle=False)

# Scoring parameter
from sklearn.metrics import mean_absolute_error


# ## Artificial Neural Network (ANN):
# - We use keras with Tensorflow backend here
# - The ANN we considered are baseline, small, deeper, custom
# - We use epoch (training round) = 30

# In[11]:


# This list will contain ANN models
nn_models = []

try:

    r,c = XY_train.shape
    #Import libraries for deep learning
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU

    # define baseline model
    def baseline(v):
        # create model
        model = Sequential()
        model.add(Dense(v*(c-1), input_dim=v*(c-1), kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    # define smaller model
    def smaller(v):
        # create model
        model = Sequential()
        model.add(Dense(v*(c-1)//2, input_dim=v*(c-1), kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='relu'))
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    # define deeper model
    def deeper(v):
        # create model
        model = Sequential()
        model.add(Dense(v*(c-1), input_dim=v*(c-1), kernel_initializer='normal', activation='relu'))
        model.add(Dense(v*(c-1)//2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='relu'))
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    # custom neural net
    def custom(v):
        model = Sequential()

        model.add(Dense(250, input_dim = c-1, kernel_initializer = 'normal'))
        model.add(Dense(100, kernel_initializer = 'normal', activation='relu'))
        model.add(Dense(50, kernel_initializer = 'normal', activation='relu'))

        model.add(Dense(1, kernel_initializer = 'normal', activation='relu'))
        model.compile(loss = 'mean_absolute_error', optimizer = 'adam')
        return(model)

    est_list = [('MLP',baseline),('smaller',smaller),('deeper',deeper),('custom', custom)]

    for name, est in est_list:

        temp = {}
        temp['name'] = name
        temp['model'] = KerasRegressor(build_fn=est, v=1, nb_epoch=30, verbose=0)

        nn_models.append(temp)

except ModuleNotFoundError:
    print('Tensorflow is not installed with GUP support')


# ## Tuning XGBoost hyperparameters:
# 
# During this step, we fist find the best n_estimators for different max_depth for max_depth = [5,6,7,8].
# 
# - max_depth
# - n_estimators

# In[12]:


xgb_dn_models = []

try:
    from xgboost import XGBRegressor
    
    dn_list = [
        (8,[230,240,250]),
        (7,[280,300,320]),
        (6,[380,400,420]),
        (5,[760,780,800])
    ]

    for d, n_list in dn_list:
        for n in n_list:
            
            model = XGBRegressor(n_estimators=n, seed=seed, tree_method='gpu_hist', max_depth=d, gamma=3, min_child_weight=3, learning_rate=0.09)

            temp = {}
            temp['name'] = "XGB-d" + str(d) + "-n" + str(n)
            temp['model'] = model

            xgb_dn_models.append(temp)

except ModuleNotFoundError:
    print('XGBoost is not installed with GUP support')


# ## Tuning XGBoost hyperparameters:
# 
# For different max_depth, we tune the value of gamma parameter.
# 
# - gamma

# In[13]:


xgb_g_models = []
try:
    from xgboost import XGBRegressor
    
    gamma_list = np.array([0, 1, 3])
    
    dn_list = [
        (8,240),
        (7,280),
        (6,400),
        (5,780)
    ]

    for d,n in dn_list:
        for gamma in gamma_list:
            
            model = XGBRegressor(n_estimators=n, seed=seed, tree_method='gpu_hist', max_depth=d, min_child_weight=3, gamma=gamma, learning_rate=0.09)

            temp = {}
            temp['name'] = "XGB-d" + str(d) + "-n" + str(n) + "-g" + str(gamma)
            temp['model'] = model

            xgb_g_models.append(temp)
except ModuleNotFoundError:
    print('XGBoost is not installed with GUP support')


# ## Tuning XGBoost hyperparameters:
# 
# For different max_depth, we tune the value of min_child_weight parameter.
# 
# - min_child_weight

# In[14]:


xgb_mcw_models = []
try:
    from xgboost import XGBRegressor

    mcw_list = np.array([2, 3, 4, 5])
    
    dng_list = [
        (8,240,3),
        (7,280,0),
        (6,400,3),
        (5,780,0)
    ]

    for d,n,g in dng_list:
        for mcw in mcw_list:
            
            model = XGBRegressor(n_estimators=n, seed=seed, tree_method='gpu_hist', max_depth=d, gamma=g, min_child_weight=mcw, learning_rate=0.09)

            temp = {}
            temp['name'] = "XGB-d" + str(d) + "-mcw" + str(mcw)
            temp['model'] = model

            xgb_mcw_models.append(temp)

except ModuleNotFoundError:
    print('XGBoost is not installed with GUP support')


# ## Tuning XGBoost hyperparameters:
# 
# For different max_depth, we tune the value of learning_rate parameter.
# 
# - learning rate

# In[15]:


xgb_lr_models = []
try:
    from xgboost import XGBRegressor
    
    lr_list = np.array([0.08, 0.09, 0.1])
    
    dng_list = [
        (8,240,3,5),
        (7,280,0,5),
        (6,400,3,4),
        (5,780,0,5)
    ]

    for d,n,g,mcw in dng_list:
        for lr in lr_list:
            
            model = XGBRegressor(n_estimators=n, seed=seed, tree_method='gpu_hist', max_depth=d, gamma=g, min_child_weight=mcw, learning_rate=lr)

            temp = {}
            temp['name'] = "XGB-d" + str(d) + "-lr" + str(lr)
            temp['model'] = model

            xgb_lr_models.append(temp)
except ModuleNotFoundError:
    print('XGBoost is not installed with GUP support')


# ## Add one more model for depth = 4:

# In[16]:


xgb_test_models = []
try:
    from xgboost import XGBRegressor
    
    lr_list = np.array([0.08, 0.09])
    
    dng_list = [
        (4,2000,3,3)
    ]

    for d,n,g,mcw in dng_list:
        for lr in lr_list:
            #Set the base model
            model = XGBRegressor(n_estimators=n, seed=seed, tree_method='gpu_hist', max_depth=d, gamma=g, min_child_weight=mcw, learning_rate=lr)

            temp = {}
            temp['name'] = "XGB-d" + str(d) + "-lr" + str(lr)
            temp['model'] = model

            xgb_test_models.append(temp)
except ModuleNotFoundError:
    print('XGBoost is not installed with GUP support')


# ## Find best models:
# - Run all models to find the one with smallest MAE for different max_depths.

# In[17]:


import pickle

all_models = [
    ('model_result_nn', nn_models),
    ('model_result_dn', xgb_dn_models),
    ('model_result_g', xgb_g_models),
    ('model_result_mcw', xgb_mcw_models),
    ('model_result_lr', xgb_lr_models),
    ('model_result_d4', xgb_test_models)
]

all_model_results = {}

for file_name, models in all_models:
    try:
        with open('../result/' + file_name, "rb") as f:
            model_result = pickle.load(f)
            all_model_results.update(model_result)

        for name, model_dict in model_result.items():
            print(name + " %s" % model_dict['avg_mean'])
    except FileNotFoundError:
        model_result = {}

        for d in models:

            model = d['model']
            name = d['name']

            model_result[name] = {}
            model_result[name]['pred'] = []
            model_result[name]['mean'] = []

            print("executing " + name)
            for i, (train_idx, val_idx) in enumerate(kf.split(XY_train)):
                print(i)
                X_train = XY_train.iloc[train_idx,:-1]
                X_val = XY_train.iloc[val_idx,:-1]
                Y_train = XY_train.iloc[train_idx,-1]
                Y_val = XY_train.iloc[val_idx,-1]
                model.fit(X_train,Y_train)

                pred = getPred(model, X_val)


                model_result[name]['pred'].append(pred)
                result = mean_absolute_error(np.exp(scaler_y.inverse_transform(Y_val)) - best_shift, np.exp(scaler_y.inverse_transform(pred)) - best_shift)
                model_result[name]['mean'].append(result)

            mean = np.mean(model_result[name]['mean'])
            print(name + " %s" % mean)
            model_result[name]['avg_mean'] = mean

        with open('../result/' + file_name, "wb") as f:
            pickle.dump(model_result, f)
            all_model_results.update(model_result)


# ## Perform Stacking:
# - A method to conbine predictions of multiple models

# In[18]:


import pickle

model_used = ['XGB-d8-lr0.08', 'XGB-d7-lr0.08', 'XGB-d6-lr0.08', 'XGB-d5-lr0.08', 'XGB-d4-lr0.08']

np.random.seed(seed)

minimum = 2000

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

preds = np.array([all_model_results[name]['pred'] for name in model_used])

mae = []
X_ensem = None
Y_ensem = None
for i, (train_idx, val_idx) in enumerate(kf.split(XY_train)):
    X_train = XY_train.iloc[train_idx,:-1]
    X_val = XY_train.iloc[val_idx,:-1]
    Y_train = XY_train.iloc[train_idx,-1]
    Y_val = XY_train.iloc[val_idx,-1]

    pred = np.array([list(preds[a][i]) for a in range(5)]).T

    if X_ensem is None:
        X_ensem = pred
        Y_ensem = Y_val
    else:
        X_ensem = np.concatenate((X_ensem, pred), axis=0)
        Y_ensem = np.concatenate((Y_ensem, Y_val), axis=0)

try:
    with open('../result/' + "ensemble_model", "rb") as f:
        ensemble_model = pickle.load(f)
except FileNotFoundError:
    ensemble_model = SVR(C=1)
    print("fitting")
    print(X_ensem.shape)
    print(Y_ensem.shape)
    ensemble_model.fit(X_ensem, Y_ensem)
    print("fitting done")
    with open('../result/' + "ensemble_model", "wb") as f:
        pickle.dump(ensemble_model, f)

print("predicting")
pred = ensemble_model.predict(X_ensem)
result = mean_absolute_error(np.exp(scaler_y.inverse_transform(Y_ensem)) - best_shift, np.exp(scaler_y.inverse_transform(pred)) - best_shift)
print('result: ' + str(result))


# ## Make Predictions:

# In[19]:


import pickle

try:
    with open('../result/' + "predictions", "rb") as f:
        predictions = pickle.load(f)
except FileNotFoundError:
    try:
        from xgboost import XGBRegressor
        X = XY_train.iloc[:,:-1]
        Y = XY_train.iloc[:,-1:]

        m1 = XGBRegressor(n_estimators=240,seed=seed, tree_method='gpu_hist',
                                  max_depth=8,
                                  gamma=3,
                                  min_child_weight=5,
                                  learning_rate=0.08)

        m2 = XGBRegressor(n_estimators=280,seed=seed, tree_method='gpu_hist',
                                  max_depth=7,
                                  gamma=0,
                                  min_child_weight=5,
                                  learning_rate=0.08)

        m3 = XGBRegressor(n_estimators=400,seed=seed, tree_method='gpu_hist',
                                  max_depth=6,
                                  gamma=3,
                                  min_child_weight=4,
                                  learning_rate=0.08)

        m4 = XGBRegressor(n_estimators=780,seed=seed, tree_method='gpu_hist',
                                  max_depth=5,
                                  gamma=0,
                                  min_child_weight=5,
                                  learning_rate=0.08)

        m5 = XGBRegressor(n_estimators=2000,seed=seed, tree_method='gpu_hist',
                                  max_depth=4,
                                  gamma=3,
                                  min_child_weight=3,
                                  learning_rate=0.08)

        m1.fit(X,Y)
        pred1 = getPred(m1, X_test)[:, None]
        print(pred1.shape)
        print("done fit 1")
        m2.fit(X,Y)
        pred2 = getPred(m2, X_test)[:, None]
        print("done fit 2")
        m3.fit(X,Y)
        pred3 = getPred(m3, X_test)[:, None]
        print("done fit 3")
        m4.fit(X,Y)
        pred4 = getPred(m4, X_test)[:, None]
        print("done fit 4")
        m5.fit(X,Y)
        pred5 = getPred(m5, X_test)[:, None]
        print("done fit 5")

        predictions = np.concatenate((pred1, pred2, pred3, pred4, pred5), axis=1)

        with open('../result/' + "predictions", "wb") as f:
            pickle.dump(predictions, f)
            
    except ModuleNotFoundError:
        print('XGBoost is not installed with GUP support')

print(predictions.shape)
predictions = ensemble_model.predict(predictions)
predictions = np.exp(scaler_y.inverse_transform(predictions)) - best_shift

# Write submissions to output file in the correct format
with open("submission.csv", "w") as subfile:
    subfile.write("id,loss\n")
    for i, pred in enumerate(list(predictions)):
        subfile.write("%s,%s\n"%(ID[i],pred))

print("Done")

