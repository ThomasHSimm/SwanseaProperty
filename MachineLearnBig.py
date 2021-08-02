def KNN_combo(colchoice,SaveFilename):
## colchoice 0=Propertydetails, 1=Location, 2=Combined
    
    import pandas as pd
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
        
    from sklearn.model_selection import GridSearchCV
    from sklearn import neighbors
    
    #import K NN libraries
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn import metrics
    
    from sklearn.metrics import mean_absolute_error
   
    import pickle
    
    import copy


    #load data
    df=pd.read_csv('SwanseaCombo_WivPricesWiv4sq.csv')
    
    #some adjustments
    df=df.drop(columns='Unnamed: 0')
    df=df.set_index('GEOGRAPHYCODE')
    df=df.drop(columns='NoOfPpl')

    #just one value for flats
    df['PcFlats']=df[['PcFlatpurposebuilt','PcFlatofHse','PcFlatCommercial']].sum(axis=1)

    
    #set columns to use
    if colchoice==1:
        colchoice=['Price', 'beach', 'park', 'sports', 'marina',
       'supermarket', 'foodstore', 'restaurant', 'takeaway', 'bar',
       'nightclub', 'school', 'university', 'shoppincentre', 'pawnshop',
       'betting', 'postoffice', 'doctor', 'hospital', 'transport', 'religion',
       'hotel', 'waste', 'office', 'industry']
    elif colchoice==0:
        colchoice=['Area_hect','Density_ppl_hect', 'RoomsPerHouse', 'BedroomsPerHouse',
       'PcNoCentralHeat',  'PcDetached', 'PcSemiD', 'PcTerraced',
       'PcFlats','PcMobile','Price']
    elif colchoice==2:
        colchoice=['Area_hect','Density_ppl_hect', 'RoomsPerHouse', 'BedroomsPerHouse',
       'PcNoCentralHeat',  'PcDetached', 'PcSemiD', 'PcTerraced',
       'PcFlats','PcMobile','beach', 'park', 'sports', 'marina',
       'supermarket', 'foodstore', 'restaurant', 'takeaway', 'bar',
       'nightclub', 'school', 'university', 'shoppincentre', 'pawnshop',
       'betting', 'postoffice', 'doctor', 'hospital', 'transport', 'religion',
       'hotel', 'waste', 'office', 'industry','Price']
    elif colchoice==3:
        colchoice=['Latitude','Longitude','Price']

    df=df[colchoice]
    
    # Normalise the data
    dfinit=copy.copy(df)#create a copy for later
    
    normQ=1#normalise the data?
    if normQ==1:
        df_=(df-df.min())/(df.max()-df.min())
        dfmin=df.min()
        dfmax=df.max()
#         df_*(dfmax-dfmin)+dfmin
    else:
        df_=df

    # get X and y values
    y=dfinit['Price']
    X = df_.drop(columns='Price')
    
    # Model Stuff
    #split into test and training data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=26)
    #check split
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)


    # investigate different number of neighbours
    XX=15
    acc=np.zeros(XX)
    acc2=np.zeros(XX)
    for k in range (1, XX+1):
        neigh = KNeighborsRegressor(n_neighbors = k).fit(X_train,y_train)
        yhat = neigh.predict(X_test)
        acc[k-1]=mean_absolute_error(y_test, yhat)
        acc2[k-1]=metrics.r2_score(y_test, yhat)
    
    #plot neighbours data with Jacc score
    plt.plot(range(0,XX),acc/max(acc))
    plt.plot(range(0,XX),acc2)
    plt.ylabel('RMS error')
    plt.xlabel('Number of neighbours (K)')
    plt.xticks(range(0,XX,1))
    # plt.xlim([0, 20])
    # plt.ylim([20000, 31000])
    plt.grid()
    # plt.yscale('log')
    plt.show()
    
    #get best automatically 
    params = {'n_neighbors':[2,3,4,5,6,7,8,9,10, 15, 20, 25, 30, 35, 40, 50, 60, 70]}
    knn = neighbors.KNeighborsRegressor()

    model = GridSearchCV(knn, params, cv=10)
    model.fit(X_train,y_train)
    n_neigh_use=( model.best_params_ )['n_neighbors']
    print('using n_neighbors as  ',n_neigh_use)
    
    # fit the model
    neigh = KNeighborsRegressor(n_neighbors = n_neigh_use).fit(X_train,y_train)

    # predict with test data
    yhat = neigh.predict(X_test)

    #what is the error
    accAll=mean_absolute_error(y_test, yhat)
    print('error, ',accAll)
    from sklearn.metrics import explained_variance_score
    print('Variance score: ',explained_variance_score(y_test, yhat))
    
    # Convert results ready for saving
    
    yhat = neigh.predict(X)
    
    dfPred=copy.copy(X)
    
    dfPred['Predicted Price']=(yhat/1000)
    dfPred['Predicted Price Difference']=dfPred['Predicted Price'].values-dfinit['Price'].values/1000

    dfPred.to_csv(SaveFilename)
    
    
    
    # Saving the objects:
    fname=SaveFilename.split('.')[0]+'.pkl'
    with open(fname, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([neigh, dfPred, dfmin, dfmax], f)
    
    # Getting back the objects:
#    with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
#        obj0, obj1, obj2 = pickle.load(f)
        
######################################################




######################################################    
def RandomForest_combo(colchoice,SaveFilename):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pickle
    
    import copy
    
    #load data
    df=pd.read_csv('SwanseaCombo_WivPricesWiv4sq.csv')
    
    #some adjustments
    df=df.drop(columns='Unnamed: 0')
    df=df.set_index('GEOGRAPHYCODE')
    df=df.drop(columns='NoOfPpl')

    #just one value for flats
    df['PcFlats']=df[['PcFlatpurposebuilt','PcFlatofHse','PcFlatCommercial']].sum(axis=1)

    
    #set columns to use
    if colchoice==1:
        colchoice=['Price', 'beach', 'park', 'sports', 'marina',
       'supermarket', 'foodstore', 'restaurant', 'takeaway', 'bar',
       'nightclub',  'shoppincentre', 'postoffice', 'doctor',
        'betting',   'transport', 'religion',
        'hotel', 'waste', 'office', 'industry','pawnshop','hospital','school', 'university']
    elif colchoice==0:
        colchoice=['Area_hect','Density_ppl_hect', 'RoomsPerHouse', 'BedroomsPerHouse',
       'PcNoCentralHeat',  'PcDetached', 'PcSemiD', 'PcTerraced',
       'PcFlats','PcMobile','Price']
    elif colchoice==2:
        colchoice=['Area_hect','Density_ppl_hect', 'RoomsPerHouse', 'BedroomsPerHouse',
       'PcNoCentralHeat',  'PcDetached', 'PcSemiD', 'PcTerraced',
       'PcFlats','PcMobile','beach', 'park', 'sports', 'marina',
       'supermarket', 'foodstore', 'restaurant', 'takeaway', 'bar',
       'nightclub', 'school', 'university', 'shoppincentre', 'pawnshop',
       'betting', 'postoffice', 'doctor', 'hospital', 'transport', 'religion',
       'hotel', 'waste', 'office', 'industry','Price']
    elif colchoice==3:
        colchoice=['Latitude','Longitude','Price']
        
    df=df[colchoice]
    
    # Normalise the data
    dfinit=copy.copy(df)
    
    normQ=1#normalise the data?
    if normQ==1:
        df_=(df-df.min())/(df.max()-df.min())
        dfmin=df.min()
        dfmax=df.max()
#         df_*(dfmax-dfmin)+dfmin
    else:
        df_=copy.copy(df)

    # get X and y values
    y=dfinit['Price']
    X = df_.drop(columns='Price')
    
    # Model Stuff
    
    #split into test and training data
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=26)
    #check split
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)
    
    #Model Stuff
    def get_score(n_estimators):
        my_pipeline = Pipeline(steps=[
            ('preprocessor', SimpleImputer()),
            ('model', RandomForestRegressor(n_estimators, random_state=0))
        ])
        scores = -1 * cross_val_score(my_pipeline, X, y,
                                      cv=3,
                                      scoring='neg_mean_absolute_error')
        return scores.mean()

    # find the best n_estimator

    results = {}
    for i in range(1,20):
        results[30*i] = get_score(30*i)
        
    n_estimators_best = min(results, key=results.get)
    
    print('Using n_estimators_best as: ', n_estimators_best)
    
    plt.plot(list(results.keys()), list(results.values()))
    plt.xlabel('n estimators')
    plt.ylabel('Mean absolute error')
    plt.show()
    
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=n_estimators_best,
                                                              random_state=0))
                             ])
    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    
    print("MAE scores:\n", scores/1000)
    print("Average MAE score (across experiments):")
    print(scores.mean()/1000)
    
    from sklearn.metrics import mean_absolute_error
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                  ('model', RandomForestRegressor(n_estimators=90,
                                                                  random_state=0))
                                 ])
    
    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, y_train)
    
    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_test)
    
    # Evaluate the model
    score = mean_absolute_error(y_test, preds)
    print('MAE:', score)
    from sklearn.metrics import explained_variance_score
    print('Variance score: ',explained_variance_score(y_test, preds))
    
    # Convert results ready for saving
    
    yhat = my_pipeline.predict(X)
    
    dfPred=copy.copy(X)
    
    dfPred['Predicted Price']=(yhat/1000)
    dfPred['Predicted Price Difference']=dfPred['Predicted Price'].values-dfinit['Price'].values/1000

    dfPred.to_csv(SaveFilename)
    
    
    # Saving the objects:
    fname=SaveFilename.split('.')[0]+'.pkl'
    with open(fname, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([my_pipeline, dfPred, dfmin, dfmax], f)
    
######################################################




######################################################

def EffectOfRoomSize(filename,typa,geocode):
    
    ## get 3 nearest places
    
    # import pandas as pd
    import pickle
    import numpy as np
    import copy
    
    # import matplotlib.pyplot as plt
    
    fname=filename.split('.')[0]+'.pkl'
    
    #load data
    if typa=='KNN':
        with open(fname,'rb') as f:  # Python 3: open(..., 'rb')
            neigh, dfPred, dfmin, dfmax = pickle.load(f)
    elif typa=='RF':
        with open(fname,'rb') as f:  # Python 3: open(..., 'rb')
            my_pipeline, dfPred, dfmin, dfmax = pickle.load(f)
        
    
    
    X2=copy.copy(dfPred)
    X2=X2.drop(columns=['Predicted Price','Predicted Price Difference'])   
    
    numberofneighs=4
    if numberofneighs==1:
        print('make sure got rid of mean in try statement')
    geocode=getNearest(geocode,numberofneighs)
    
    # print(X2)
    
    try:    
        Xmean=X2.loc[geocode].mean()
        ##if geocode has one value get rid of meAN above
        print('using location value',geocode)
    except:
        Xmean=X2.mean()
        print('using mean value')
        
    # print(Xmean)
    
    
    X2=X2.iloc[0:20]

    X2.iloc[0:20,:]=Xmean
    
    newRooms=(np.linspace(-.2,1.1,20))
    
    X2['RoomsPerHouse']=newRooms
    
    newRooms=(np.linspace(0,1.1,20))
    # X2['BedroomsPerHouse']=newRooms
    
    if typa=='KNN':
        Ymeanpred= neigh.predict(X2)#X.iloc[0].values)
    else:
        Ymeanpred= my_pipeline.predict(X2)#X.iloc[0].values)
    
    # print(Ymeanpred)
    Xback=X2*(dfmax[0:-1]-dfmin[0:-1])+dfmin[0:-1]
    
    # print(Xback)
    
    for ii in range(0,20):
        X2.iloc[ii]=X2.iloc[ii]*(dfmax[0:-1]-dfmin[0:-1])+dfmin[0:-1]
    
    
    # plt.scatter(X2['RoomsPerHouse'],Ymeanpred/1000,marker='+',s=100) 
    # plt.xlabel('Number of rooms')
    # plt.ylabel('House Price (£ 000s)')
    # plt.grid()

    priceDiff=Ymeanpred.max()/1000-Ymeanpred.min()/1000
    print('Range prices rooms',priceDiff)
    return X2['RoomsPerHouse'],Ymeanpred/1000, priceDiff


######################################################




######################################################

def EffectOfRoomSizeCombo(filename,typa):
    
    ## get 3 nearest places
    
    # import pandas as pd
    import pickle
    import numpy as np
    #import matplotlib.pyplot as plt
    import copy
    
    fname=filename.split('.')[0]+'.pkl'
    
    #load data
    if typa=='KNN':
        with open(fname,'rb') as f:  # Python 3: open(..., 'rb')
            neigh, dfPred, dfmin, dfmax = pickle.load(f)
    elif typa=='RF':
        with open(fname,'rb') as f:  # Python 3: open(..., 'rb')
            my_pipeline, dfPred, dfmin, dfmax = pickle.load(f)
            neigh=my_pipeline
        
    X2=copy.copy(dfPred)
    X2=X2.drop(columns=['Predicted Price','Predicted Price Difference'])   
    
    pdiff=np.zeros(len(X2))
    ii=0
    for geoC in X2.index:
        pdiff[ii]=  inner(X2,geoC,neigh)
        ii+=1
        
    X2.loc[:,'RoomEffect']=pdiff
    
    return X2
   



def inner(X2,geocode,neigh):
    import numpy as np
    import copy
    
    numberofneighs=4
    geocode=getNearest(geocode,numberofneighs)
    

    Xmean=copy.copy(X2.loc[geocode,:].mean())
                    

    X3=copy.copy(X2.iloc[0:20,:])

    X3.iloc[0:20,:]=Xmean
    
    newRooms=(np.linspace(-.2,1.1,20))
    
    X3.loc[:,'RoomsPerHouse']=newRooms
    
    newRooms=(np.linspace(0,1.1,20))
    # X2['BedroomsPerHouse']=newRooms
    
    
    ## PREDICTIONS!!!!
    
    Ymeanpred= neigh.predict(X3)#X.iloc[0].values)
    
    ##
            
    priceDiff=Ymeanpred.max()/1000-Ymeanpred.min()/1000
    
    return priceDiff

######################################################




######################################################


def getNearest(geocode,nearestNo):
    import pandas as pd
    #input geocode of a district / how many neighbours
    #output list of geocodes
    df=pd.read_csv('SwanseaCombo_WivPricesWiv4sq.csv',usecols=['GEOGRAPHYCODE','Latitude','Longitude','Price'])
    df=df.set_index('GEOGRAPHYCODE')
    
    longTarget=df.loc[geocode].Longitude
    latTarget=df.loc[geocode].Latitude
    
    latDiff=(111*(df.Latitude - latTarget))**2
    longDiff=(85*(df.Longitude - longTarget))**2
    
    distTarget = (abs(latDiff.values - longDiff.values))**0.5
    
    df['DistTarget']=distTarget
    df=df.sort_values(by=['DistTarget'])
    geonew=df.iloc[0:5].index
    return geonew


######################################################




######################################################

def locationeffect(filename,typa):
       
    
    import pickle
    
    
    import copy 
    
    fname=filename.split('.')[0]+'.pkl'
    
    #load data
    if typa=='KNN':
        with open(fname,'rb') as f:  # Python 3: open(..., 'rb')
            neigh, dfPred, dfmin, dfmax = pickle.load(f)
    elif typa=='RF':
        with open(fname,'rb') as f:  # Python 3: open(..., 'rb')
            my_pipeline, dfPred, dfmin, dfmax = pickle.load(f)
        
    
    
    X2=copy.copy(dfPred)
    X2=X2.drop(columns=['Predicted Price','Predicted Price Difference'])   
    
    loccols=['beach', 'park', 'sports', 'marina', 'supermarket',
       'foodstore', 'restaurant', 'takeaway', 'bar', 'nightclub', 'school',
       'university', 'shoppincentre', 'pawnshop', 'betting', 'postoffice',
       'doctor', 'hospital', 'transport', 'religion', 'hotel', 'waste',
       'office', 'industry']
    
    X3=copy.copy(X2)
    for geoC in X2.index:
        #for each region change no location values
        for loca in loccols:
            X2.loc[:,loca].loc[geoC]=X3.loc[:,loca].mean()
            
        
    
    if typa=='KNN':
        Ymeanpred= neigh.predict(X2)#X.iloc[0].values)
    else:
        Ymeanpred= my_pipeline.predict(X2)#X.iloc[0].values)
    
    dfPred['LocationEffect']=dfPred['Predicted Price']-Ymeanpred/1000
        
    
    return dfPred     


####################################################################################

###################################################################################
def RandomForest_LocationDifferent(fname):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pickle
    import copy
    
    #load data for location
    df=pd.read_csv('SwanseaCombo_WivPricesWiv4sq.csv',usecols=['GEOGRAPHYCODE','Latitude','Longitude','Price'])
        
    #load data for analysis
    
    fname=fname.split('.')[0]+'.pkl'
        # Getting back the objects:
    with open(fname,'rb') as f:  # Python 3: open(..., 'rb')
       swansea_grouped, swansea_venues = pickle.load(f)
    
    #MErge the data
    #merge data together
    df=pd.merge(df,swansea_grouped,left_on='GEOGRAPHYCODE',right_on='Neighborhood')
    df=df.set_index('GEOGRAPHYCODE')
    df=df.drop(columns='Neighborhood')
    dfinit=copy.copy(df)
    
    df=df.drop(columns=['Latitude','Longitude'])
    
    
    # Normalise the data
    
    
    normQ=1#normalise the data?
    if normQ==1:
        df_=(df-df.min())/(df.max()-df.min())
        dfmin=df.min()
        dfmax=df.max()
#         df_*(dfmax-dfmin)+dfmin
    else:
        df_=copy.copy(df)

    # get X and y values
    y=dfinit['Price']
    X = df_.drop(columns='Price')
    
    # Model Stuff
    
    #split into test and training data
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=26)
    #check split
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)
    
    #Model Stuff
    def get_score(n_estimators):
        my_pipeline = Pipeline(steps=[
            ('preprocessor', SimpleImputer()),
            ('model', RandomForestRegressor(n_estimators, random_state=0))
        ])
        scores = -1 * cross_val_score(my_pipeline, X, y,
                                      cv=3,
                                      scoring='neg_mean_absolute_error')
        return scores.mean()

    # find the best n_estimator

    results = {}
    for i in range(1,20):
        results[30*i] = get_score(30*i)
        
    n_estimators_best = min(results, key=results.get)
    
    print('Using n_estimators_best as: ', n_estimators_best)
    
    plt.plot(list(results.keys()), list(results.values()))
    plt.xlabel('n estimators')
    plt.ylabel('Mean absolute error')
    plt.show()
    
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=n_estimators_best,
                                                              random_state=0))
                             ])
    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    
    print("MAE scores:\n", scores/1000)
    print("Average MAE score (across experiments):")
    print(scores.mean()/1000)
    
    from sklearn.metrics import mean_absolute_error
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                  ('model', RandomForestRegressor(n_estimators=90,
                                                                  random_state=0))
                                 ])
    
    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, y_train)
    
    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_test)
    
    # Evaluate the model
    score = mean_absolute_error(y_test, preds)
    print('MAE:', score)
    from sklearn.metrics import explained_variance_score
    print('Variance score: ',explained_variance_score(y_test, preds))
    
    # Convert results ready for saving
    
    yhat = my_pipeline.predict(X)
    
    dfPred=copy.copy(X)
    
    dfPred['Predicted Price']=(yhat/1000)
    dfPred['Predicted Price Difference']=dfPred['Predicted Price'].values-dfinit['Price'].values/1000

    dfPred.to_csv(fname.split('.')[0]+'.csv')
    
    
    # Saving the objects:
    fname=fname.split('.')[0]+'.pkl'
    with open(fname, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([my_pipeline, dfPred, dfmin, dfmax], f)
    
######################################################
