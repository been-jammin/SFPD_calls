#!/usr/bin/env python
# coding: utf-8

# first let's import all the packages we'll need for this section. they are all standard steps from the sklearn library. except for category_encoders, which is an extension package of sklearn. it has an ordinal encoder that we will experiment with later on.

# In[16]:


import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import LinearSVR
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
 


# i like having our base dataset named "dataset", so let's name it that. which we will do by reading the stored variable from the other notebook. when i'm working in jupyter lab, i can simply switch the kernel of this notebook to use the kernel of the notebook where i imported the data. but if you are following along and don't have that functionality, you can always just add %store results_df to the end of the prior notebook, and %store -r results_df to load the variable back into kernel memory
# 
# 

# In[17]:


# %store -r results_df
dataset = results_df


# # data cleaning

# first, let's take a look at what we are considering to be our 2 predicted quantities, "response duration"  and "travel time" (both defined in prior notebook) and check them for NaNs using Pandas methods isnull() and value_counts()

# In[18]:


check = dataset['response duration'].isnull()
print(check.value_counts())


# In[19]:


check = dataset['travel time'].isnull()
temp = dataset['travel time'].value_counts()
print(check.value_counts())


# looks like there are some NaNs. probably of the same observations. we probably will have enough data that we can ignore/remove these observations, but i find it more elegant to replace them with a value that is closer to the truth. so instead let's calculate a mean and replace the NaNs with that value. recall that when the course filtered to times <15min the "response duration" distribution looked pretty gaussian, so adding more frequency to the mean shouldn't change the model. but we can try it both ways. 

# In[20]:


goodMean = dataset['response duration'].mean(skipna = True)
dataset['response duration'] = dataset['response duration'].fillna(goodMean)


# In[21]:


goodMean = dataset['travel time'].mean(skipna = True)
dataset['travel time'] = dataset['travel time'].fillna(goodMean)


# In[22]:


dataset['response duration'].isnull().value_counts()


# even though earlier we discussed performing the analyis on both "travel time" and "response duration", we need to start focusing so we're not doing 2 different analyses at once. so in order to keep our distributions simple, let's focus on 'response duration' whose values are between 0 and 15 minutes.

# In[23]:


dataset_raw = dataset
dataset = dataset_raw[dataset_raw['response duration'].between(0,15)]
dataset.columns


# what i realized after pouring through this data a few times is there can be several row enteries associated with the same incident. these rows will have the same "Received_dttm", but perhaps different unit_types or response times, as maybe several different types of units respond to the same incident but at different speeds. so what we will really be looking at is response speed per unit. anyway, as a way of making sure we don't have any duplicates of the data, i have also included the rowid, which we can run value_counts() on to make sure there are no duplicates.

# In[24]:


dataset.shape


# In[76]:


dataset['rowid'].value_counts()


# turns out there are some duplicates. they must have been in the dataset from the beginning. regardless, we can drop them easily.

# In[77]:


dataset = dataset.drop_duplicates()
print(dataset.shape)


# # Data Exploration (Visualizations)

# at this point, the course goes on to immedately start the train/test split, encoding, and fitting a model. but one of the gravest errors in machine learning is building a model too early, before you've gotten really familiar with the data. so let's go outside the course and do some of that now. let's see if we can spot any correlations between the features of our observations, and the response duration, which is the variable we are trying to predict. presumably, because the course chose to track these features, there should be some. 
# after all, instinctively, we would think that things like call type, number of alarms, and priority should all have some kind of impact on the response duration. to do this, let's plot the response duration against the elements of all the features. for simplicity, we will just collapse the data by using a pandas pivot table. this will just take an average of all response durations for each feature element

# In[78]:


dataset.columns


# In[79]:


def plotAgainst(dataset,column, y_axis):
    dataset_piv = pd.pivot_table(dataset, index = column)
    dataset_piv = dataset_piv.reset_index()
    dataset_piv = dataset_piv.sort_values(by = y_axis)
    plt.scatter(dataset_piv[column],dataset_piv[y_axis])
    label = (y_axis, ' vs. ', column)
    plt.title(label)


# In[80]:


dataset_columns = dataset.columns


# In[81]:


cols_to_plot = ['call_Type', 'fire_prevention_district','neighborhoods_analysis_boundaries', 'number_of_alarms', 'original_priority', 'priority','unit_type']

for i in range(len(cols_to_plot)):
    plt.figure()
    plotAgainst(dataset,cols_to_plot[i],'response duration') 
    plt.show()


# In[ ]:





# so we've learned a few things from this, before even building models. all the columns that the courses uses to predict the response duration are correlated to it in some way. i'm a bit surprised that neighborhood and district seem to have a strong correlation. i would think that the fire department should be able to respond with equal speed to all parts of the city...isn't that kind of the point of the fire department? anyway, it will be interesting to see how that impacts the prediction. also, unit type seems to have an impact. some quick research reveals that "unit type" refers to the style of vehicle used to respond to the incident. i guess this is in some way related to the "call type", so the FD can make sure they have the appropriate equipment to respond to the incident.
# 
# anyway, given that we will be predicting a continuous quantity (time, in minutes), we will be using regression estimators, which require numerical inputs, which means we will need to encode our data. the course uses a straight one hot encoder for all features. but this is a possible divergence point. i would think that features like "priority" have a natural order to them. and it seems to be true somewhat. i would consider the two priority features as being more ordinal featres than categorical features. but the course treats them as categorical, so we'll follow suit.  but later on we will try with an Ordinal encoder. hence why we imported it earlier.
# 

# # Train/Test Split and one-hot encoding

# standard steps for an 80/20 train/test split

# In[129]:


X = dataset.drop(['travel time','response duration','dispatch_dttm','response_dttm', 'received_dttm', 'rowid'], axis=1)
y = dataset['response duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 

# In[130]:


ohe_train = OneHotEncoder(handle_unknown = 'ignore')
Xtrain_enc = ohe_train.fit_transform(X_train)
ohe_train.categories_


# In[ ]:





# In[131]:


ohe_test = OneHotEncoder(categories = ohe_train.categories_, handle_unknown = 'ignore')
Xtest_enc = ohe_test.fit_transform(X_test)
df_map = pd.DataFrame({'original column':X.columns,
                       'feature labels':ohe_train.categories_})


# In[132]:


ohe_test.categories_


# just to make sure everything came out the right shape...ie that the training set or test set didn't end up with more feature labels than the other

# In[133]:


print(Xtrain_enc.shape)
print(Xtest_enc.shape)


# in order to judge our model performance against a standard baseline, let's see how good our predictions might be if we just predicted the average response duration every time.

# In[134]:


avgDelay = np.full(y_test.shape, np.mean(y_train), dtype=float)
avgDelay


# In[135]:


print(r2_score(y_test,avgDelay))
print(mean_squared_error(y_test, avgDelay, squared = False))


# so comparing the response durations of the test set against a baseline, we see that on average we are off by an average of about 2 minutes.

# # create models and queue (divergence from course content)
# in the course, we only work with one estimator, the simple linear regressor. but we'll be trying a bunch. so let's create a dataframe to hold the scores of each estimator we try, for quick comparison

# In[136]:


if 'scores' not in locals():
    scores = pd.DataFrame({'model':[],
              'score':[]})
else:
    del scores
if 'rmse_scores' not in locals():
    rmse_scores = pd.DataFrame({'model':[],
              'RMSE score':[]})
else:
    del rmse_scores


# this is where the course ends. it introduces the linear regression model, fits it, shows that it has a better RMSE than the baseline, and ends.
# it is also where i will attempt to go deeper than the course and uncover some more interesting insights about the data.
# 
# the course only uses one estimator, but we should try a few to be sure we find the best one. estimators have different strengths for the nature of dataset they work on. for regression problems with our number of features and observations, we'll try all the relevant ones, with standard hyperparamters. we can tune these later on.
# 
# in the case of ridge and lasso, we can use the built-in cross-validation functionality so we know what hyperparmeters we should focus in on. feel free to turn down the verbosity to save screen space. i like it.

# In[137]:


lr = linear_model.LinearRegression(fit_intercept=True, normalize=True)
sgdr = linear_model.SGDRegressor(max_iter=1000, tol =1e-3, verbose = 3)
ridge = linear_model.Ridge(alpha = 1)
ridge_cv = linear_model.RidgeCV(alphas = [1e-1,1e0,1e1])
lasso = linear_model.Lasso(alpha = 1)
lasso_cv = linear_model.LassoCV(alphas = [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3], verbose = 3)
linSVR1 = LinearSVR(C=0.1, verbose = 3)
linSVR2 = LinearSVR(C=1, verbose = 3)
linSVR3 = LinearSVR(C=10, verbose = 3)
elNet = linear_model.ElasticNet(alpha = 1, l1_ratio = 0.5)


# a cool part of my project will be to investigate feature importances. so we'll need a dataframe to hold those. 

# 

# In[138]:



importances = pd.DataFrame({'feature':ohe_train.get_feature_names()})
importances['feature category'] = importances['feature'].str[1]


# # fit models to training set

# there may be a more elegant approach to this, like possibly using pipelines. but for now, all we want to do is try each estimator on the dataset to see which comes out the strongest. so we'll make a queue of all the estimators we created eariler, fit each one to the training data, and then make predictions and record the scores for quick comparison. 
# 
# in the course, the scoring method chosen is the (root) mean squared error method. which makes sense, as that can be interpreted as an average number of minutes that the predictor is off by. but the built-in scoring method of all the estimators we will use is the r-squared. intuitively, this makes more sense to score models when the input is continuous, not ohe. regardless, we'll use both.

# In[162]:


models = [lr,sgdr, ridge, ridge_cv, lasso, lasso_cv, linSVR1, linSVR2, linSVR3, elNet]
#models = [lr, sgdr, ridge_cv, lasso_cv, elNet]
#models = [lr, sgdr, ridge, lasso, elNet]

def fitPredictScore(modelQueue, X_train, y_train, X_test, y_test, scores, rmse_scores):
   
    for i in range(len(models)):
        print('fitting ',models[i])
        models[i].fit(X_train,y_train)    
        y_pred = models[i].predict(X_test)
        modelScore = models[i].score(X_test, y_test)
        scores = scores.append({'model':str(models[i].__class__.__name__),
                                  'score':modelScore},
                                  ignore_index = True)

        rmse_scores = rmse_scores.append({'model':str(models[i].__class__.__name__),
                                  'RMSE score':mean_squared_error(y_test, y_pred, squared = False)},
                                  ignore_index = True)
    return scores, rmse_scores


scores, rmse_scores = fitPredictScore(models,Xtrain_enc, y_train, Xtest_enc, y_test,scores=pd.DataFrame(),rmse_scores = pd.DataFrame())


# In[163]:


scores.sort_values(by='score', ascending = False)


# In[164]:


rmse_scores


# the first thing to note is that in general, these r-square values look quite poor. obviously, what is a "good" vs "bad" r-square depends on the problem and the data, but intuitively i thought these estimators could do better. 
# 
# finally, it looks like the basic linear regressor and the ridge_cv (with alpha of 0.1) had the best scores - both the highest r-square and the lowest RMSE. this is probably why the course chose to introduce just the simple linear regressor. but the good news is that using this estimator instead of just predicing the average has improved the quality of our response duration prediction, from being off by 2 minutes, to being off by 1.73 minutes. still...this is only a 14% improvement in prediction quality. and if the average response duration is ~3 minutes, and we can only predict it within ~1.73 minutes, that's not very good. hopefully we can improve with other modifications to the model.
# 
# out of curiosity, let's check the best value of alpha used in the cross-validations.
# 

# In[165]:


print(ridge_cv.alpha_)
print(lasso_cv.alpha_)


# both decently low. indicating that assigning weights to more features won't be heavily penalized. an alpha of 0 is just a linear regression. which makes sense in retrospect, recalling that our data is one-hot encoded. with 6 features, each with several labels, exploded out to 107 columns. we can expect that each of these columns will impact the prediction significantly. so if we expect all our features to contribute significantly, a basic linear regressor estimator makes sense as the strongest model. the correlations shown in the scatterplots earlier corroborate this.
# 
# a ridge estimator with alpha = 0.1 means that only slightly penalizing the weights of some features...which turns out not to help the predictibilty.

# In[ ]:





# # compile feature importances

# to examine the feature importances, i will use the permutation_importance() method, which takes an estimator, and iteratively re-fits it to the data with a new column dropped every time. the idea being that the most important features will do the greatest damage on the model when they are dropped
# 
# once the features have been assigned importances, we will append them to the dataframe of all the feature labels, so we can compare the importances given by each model

# In[166]:


if 'importances' not in locals():
    importances = pd.DataFrame({'feature':encoder.get_feature_names()})
    importances['feature category'] = importances['feature'].str[1]

goodModels = [lr, sgdr, ridge_cv, lasso_cv]
goodModels = [lr]

for i in range(len(goodModels)):
    print('calculating feature importances for: ', goodModels[i])
    feIm = permutation_importance(goodModels[i],Xtrain_enc.toarray(), y_train, n_repeats = 3)
    importances[str(goodModels[i])] = feIm.importances_mean


# In[167]:


importances


# recall that our data is one-hot encoded. so it's not really feature importances we are looking at, but feature label importances. which doesn't help us very much. but it could be cool to see which single labels have the biggest impact on response duration
# 
# so far, the linear regressor is our strongest estimator, so we'll use that estimator's feature importances to sort by

# In[168]:


importances.sort_values(by=['LinearRegression(normalize=True)'], ascending = False)[1:10]


# looks like the most important labels are the "original priority" being assigned value "A" or"B" the call_type being assigned "medic", or the unit_type being assigned "MEDIC" or "RESCUE CAPTAIN".
# 
# still, the more vital information to extract would be which feature (not feature label) has the greatest bearing on response duration. in order to do that, we take an average of the importances over each feature category. i'm aware there may be a more model-rigorous way to do this, but this should satisfy our curiosity for now.
# 
# to do this, we can have pandas create a pivot table, aggregating the data by feature category, and taking an average

# In[169]:


importances_pivot = pd.pivot_table(importances, index = 'feature category',aggfunc = 'mean')
importances_pivot


# this table is more helpful. reading from our best-performing estimator, we see that on average, the most important feature groups are 4 and 6. feature group 4 is the "original priority", and 6 is the unit_type. intuitively, this makes sense. 
# 
# # results discussion
# 
# i'm not an expert on how fire department calls are handled, but i imagine that at some point the dispatcher assigns a priority to each incident that gets called in. maybe this is assigned by a person, maybe it is assigned by an algorithm based on information that the dispatcher inputs. either way, it is logical that that would have a big impact on how quickly the fire department arrives on the scene. recall from the earlier scatterplot that we might have been able to guess this, given the strong visual correlation. also, that plot reveals the order of the original priority assignments, which seems to be (from highest to lowest) B, I, A, C, E, 3, 2. but this uncovers another problem. maybe the "original priority" is itself a output/target variable, using other features as its input. like the call_type, the neighborhood, or the number of alarms for example. in that case, another worthwhile exercise could be to remove it from the features (given that by its own right is correlated to the input features), or attempt to predict it using a different model. the same applies to the 2nd most important feature, the unit_type. to some degree, this would depend on the type of incident being reported. so maybe it too should be removed from the model to give it more versatility.
# 
# i think we can agree that some improvements can be made to the model. perhaps the greatest of which is to realize that there are 3 fields in the data that prehaps should not have been one-hot encoded. the "number of alarms" shouldn't need to be encoded at all. it's just an integer, which holds the number of alarms that the incident caused. in addition, given the discussion just presented, perhaps the two "priority" fields should be enoded not with a OHE, but with an ordinal encoder, given that the feature labels do have a natural "order". so let's try that next.

# # try an ordinal encoder

# this ordinal encoder we are using requires a list of dictionaries. it needs to be explicitly told what is the proper "ordering" of the feature labels. so we will give it that, for each column that we are applyind the encoder to

# In[170]:


ordinalMapping = [{'col':'original_priority',
                   'mapping':
                            {'B':1,
                            'I':2,
                            'A':3,
                            'C':4,
                            'E':5,
                            '3':6,
                            '2':7}},
                     {'col':'priority',
                      'mapping':{
                              'E':1,
                              '3':2,
                              '2':3}},
                      {'col':'number_of_alarms',
                       'mapping':
                           {'1':1,
                            '2':2}
                              }]
                    
                                  


# In[171]:


ordEnc = ce.OrdinalEncoder(mapping = ordinalMapping)
Xtrain_opt = ordEnc.fit_transform(X_train)
Xtrain_opt = Xtrain_opt.astype({'original_priority':'int64','priority':'int64', 'number_of_alarms':'int64'})

Xtest_opt = ordEnc.fit_transform(X_test)
Xtest_opt = Xtest_opt.astype({'original_priority':'int64','priority':'int64', 'number_of_alarms':'int64'})


# In[172]:


dataset['number_of_alarms'].value_counts()


# In[173]:


Xtrain_opt


# we still want to one hot encode all the other features, meaning we will have a combination of one-hot encoded data and ordinally encoded data. in order to transform different column groups separately, we need to use the ColumnTransformer. it's a simple tool where we just have to tell it which encoder to use on which columns
# 
# then we can use the column transformer to fit_transform the raw training and test data.

# In[174]:


from sklearn.compose import ColumnTransformer
    
ordinal_features = ['number_of_alarms','original_priority','priority']
categorical_features = ['call_Type','fire_prevention_district', 'neighborhoods_analysis_boundaries', 'unit_type']

ohe_train = OneHotEncoder(handle_unknown = 'ignore')

colTrans_train = ColumnTransformer(transformers = [

    ('cat_train',ohe_train, categorical_features),
    ('ord',ordEnc,ordinal_features)
])


# In[175]:


Xtrain_opt_enc = colTrans_train.fit_transform(Xtrain_opt)

ohe_train_categories = colTrans_train.named_transformers_['cat_train'].categories_


# In[176]:




ohe_test = OneHotEncoder(categories = ohe_train_categories, handle_unknown = 'ignore')

colTrans_test = ColumnTransformer(transformers = [
    ('cat_test',ohe_test, categorical_features),
    ('ord',ordEnc,ordinal_features)
])

Xtest_opt_enc = colTrans_test.fit_transform(Xtest_opt)


# In[177]:


print(Xtrain_opt_enc.shape)
print(Xtest_opt_enc.shape)


# before, when all our features were one hot encoded, there was no need to apply any center/scaling because after encoding, the values in the columns would have all been either 0 or 1. now that we have some integer values in there, we need to center and scale the data before we fit any models to it.
# 
# there a few options of how to scale the data, so we'll try them all
# 
# then we can fit and predict all the models in our queue as before and append their scores to the scores dataframe

# In[178]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler

mms = MinMaxScaler()
mas = MaxAbsScaler()
sts = StandardScaler(with_mean = False)

Xtrain_enc_opt_scaled = sts.fit_transform(Xtrain_opt_enc)
Xtest_enc_opt_scaled = sts.fit_transform(Xtest_opt_enc)


# In[179]:


print(Xtrain_enc_opt_scaled.shape)
print(Xtest_enc_opt_scaled.shape)
print(y_test.shape)


# In[180]:


models = [lr,sgdr,ridge_cv,lasso_cv,elNet]
models = [lr, ridge_cv]
scores, rmse_scores = fitPredictScore(models,Xtrain_enc_opt_scaled,y_train,Xtest_enc_opt_scaled, y_test, scores, rmse_scores)
    
print(scores)
print(rmse_scores)


# unfortunately, these did not help our scores. nor did they help the SVR estimators converge. this surprises me, as i would think that ordinal encoding of the appropriate features would make for a more accurate model. but i guess not.

# # Further thoughts and discussion of possible next steps
# 
# in the course, the instructors claim victory by bettering the baseline predictions by ~0.3 minutes. but that's not very satisfying for me. but it seems like we've done all we can do with the features that the course has selected. they probably did their homework and found that these fields had the strongest correlation to the response duration. i'm sure there is also a strong correlation between response duration and the distance between the incident and the fire house that dispatched the unit. as well as time-of-day. but these seem a bit too obvious to me. i would rather find a connection to something lurking beneath the surface. hence my satisfaction with finding the connections between priority and unit type.
# 
# to satisfy my curiosity, i searched for other projects that are similar to what i've attempted. and i found one that's almost exactly the same. https://medium.com/crim/predicting-the-response-times-of-firefighters-using-data-science-da79f6965f93
# 
# in this project, Mr. Pecoraro endeavours to predict response times of the montreal fire department in a very similar way. he uncovers that his response times, too, are heavily influenced by the style of unit being dispatched. and not only that, he breaks the set down into "faster units" and "slower units" and finds that they have different distributions. hence, he splits up the dataset, and uses two different models - one for each style of unit. and his goal is also to beat a predictive model that simply predicts the average response time. he uses more features, like time-of-day. which is encouraging, and means maybe we can incorporate that feature into our predictive model here. a possible flaw in his approach is that he uses all calls before 2016 as his training set, and all calls after 2016 as his test set. this ignores any personnel/protocol/procedural/equipment changes that may have occured in the MPD between those two timespans. regardless, using machine learning algorithms, he was able to improve his prediction errors. quoting, "The mean absolute error (MAE) decreased from 17.8 seconds (baseline) to 15.7 seconds (XGBoost). When evaluating the model on the slower units only, the MAE decreased from 158 seconds to 85.6 seconds. When evaluating the faster units only, the error decreased from 16.9 seconds to 15.2 seconds."
# 
# this makes me feel better about my prediction accuracy improving by 14% from baseline.
# 
# regardless, as a final effort, let's see what happens when we incorporate received_dttm (time-of-day when call was received) as a continuous input feature. 
# 

# # adding timestamp of call as a continuous input feature
# 
# first, let's check to see if there's any correlation between the date and time of the call, and the response duration. i wouldn't think there should be. and even if there is, i would think it would be cyclical with a period of days or weeks, with slower response times ocurring during rush hours or holidays.

# In[181]:




plt.figure()
plt.scatter(dataset['received_dttm'], dataset['response duration'])
plt.show()

recent_calls = dataset[dataset['received_dttm'].between('2019-07-01','2019-07-31')]
plt.figure()
plt.scatter(recent_calls['received_dttm'], recent_calls['response duration'])
plt.show()


# looks like no visible correlation, but let's try it anyway.
# 
# a quick recap of steps:
# - create new X and Y set from original dataset, but with received_dttm column included.
# - convert received_dttm column to an integer timestamp (because sklearn models can't handle python datetimes as an input)
# - use column transformer to one-hot encode all features except received_dttm (as int)
# - perform feature scaling on timestamp column, using a max-absolute scaler (only one that can handle a sparse matrix)
# - fit/predict/score the linear regressor estimator, and compare scores

# In[182]:


X2 = dataset.drop(['travel time','response duration','dispatch_dttm','response_dttm', 'rowid'], axis=1)
y = dataset['response duration']


print(X.shape)
print(X2.shape)


# In[183]:


import datetime as dt
import time as time
X2['received_dttm_int'] =X2['received_dttm'].values.astype(np.int64) // 10 ** 9
X2 = X2.drop(['received_dttm'], axis = 1)
X2.dtypes


# In[184]:


X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)


# In[185]:


categorical_features = ['call_Type','fire_prevention_district', 'neighborhoods_analysis_boundaries', 'unit_type','number_of_alarms','original_priority','priority']
continuous_features  = ['received_dttm_int']

ohe_train = OneHotEncoder(handle_unknown = 'ignore')

colTrans = ColumnTransformer(transformers = [
    ('cat',ohe_train, categorical_features)], remainder = 'passthrough')

X2_train_enc = colTrans.fit_transform(X2_train)

ohe_train_categories = colTrans.named_transformers_['cat'].categories_
ohe_train_categories


# In[ ]:





# In[186]:


ohe_test = OneHotEncoder(categories = ohe_train_categories, handle_unknown = 'ignore')

colTrans = ColumnTransformer(transformers = [
    ('cat',ohe_test, categorical_features)], remainder = 'passthrough')

X2_test_enc = colTrans.fit_transform(X2_test)

print(X2_train_enc.shape)
print(X2_test_enc.shape)


# In[187]:


mms = MinMaxScaler()
mas = MaxAbsScaler()
sts = StandardScaler(with_mean = False)

X2train_enc_opt_scaled = mas.fit_transform(X2_train_enc)
X2test_enc_opt_scaled = mas.fit_transform(X2_test_enc)


# In[188]:


models = [lr, ridge]
scores, rmse_scores = fitPredictScore(models, X2train_enc_opt_scaled, y_train, X2test_enc_opt_scaled, y_test,scores, rmse_scores)
scores.sort_values(by='score',ascending = False)


# In[ ]:





# In[ ]:





# ok so it turns out adding the timestamp of the call as a feature still doesn't help the model prediction. looks like the course really did their homework and taught the best model, which also happened to be the simplest.
# 
# but let's try and go one step further. we know that the 2nd most important feature is unit_type. and we know Mr. Pecoraro got good results when he split this group up. so let's give that a try. first, let's take a look at the correlation between unit_type and response duration. we see that "investigations" tend to take much longer than other incidents. makes sense. let's see what the distribution is of that one unit_type
# 
# 

# In[ ]:





# In[189]:


subset = dataset[dataset['unit_type']=='INVESTIGATION']
plt.hist(subset['response duration'])
subset['response duration'].value_counts()


# that's a pretty unfriendly distribution. and implies that when the "investigation" unit is sent, response times can be quite long. but maybe that's because an "investigation" get dispatched to the same incident, alongside other, higher-priority units that can respond to incidents more quickly.
# 
# given the unfriendliness of this distribution, the model may improve if we remove these "investigation" observations from the training set. but then again, there are only 20 observations like this out of a set of 100k samples. so it probably won't change the model at all.

# In[190]:


subset = dataset[dataset['unit_type']=='AIRPORT']
plt.hist(subset['response duration'])
subset['response duration'].value_counts()


# this is another interesting connection. any unit types with the value "airport", all have zero as their response duration. this is probably because these units are responding to incidents at there airport, and there is a fire station inside the airport. pulling these out of the dataset would probably improve the model. but there are only 9 observations like this in a dataset of 100k observations. so it probably won't change at all.

# well i think that's all the time i have for now. a few final thoughts before i sign off...
# 
# future ideas for a stronger model:
# - either use another model to predict the priority of the indcident from other features, or remove it from the model.
# - incorporate distance between fire stations and incident location as an input feature (would require GPS coordinates of all firehouses)
# - check distribution of all unit_types for skewness. if necessary, split into two datasets (and hence two models), with each one being a group of units. perhaps a "faster units" and "slower units" as Mr. Pecoraro has done
# - in order to achieve a better model score, perhaps we could run a clustering model on the resonse_durations and cluster them into groups of "average speed", "above average", and "below average". then run a classification model on the input features to simply predict whether the response speed will be "fast", "average", or "slow". realistically, this may not have as impactful of an application as predicting a continuous response time in minutes, but all we care about is model score, this may help.

# finally, thanks for following along with my work. if you have any questions or suggestions for improvement or new projects, please feel free to raise an issue on GitHub or email me at Benjamin.A.Cohen.90@gmail.com.
# 
# some ideas for future projects, from my hobbies:
# - prediction of "sonic coincidences" (when one song unintentionally sounds similar to another) using spotify data
# - analysis of bike makes/models/sales using Strava data
