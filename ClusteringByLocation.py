def Clustering(fname,*args):
# call fname, radius, LIMIT, section
## call with # swansea_venues = getNearbyVenues(names=df['GEOGRAPHYCODE'], latitudes=df['Latitude'],longitudes=df['Longitude'] )
    
    import pandas as pd
   
    import pickle
    
    
    i=0
    for arg in args:
        if i==0:
            radius=arg
        elif i==1:
            LIMIT=arg
        elif i==2:
            section=arg
        i=i+1
    
    try:
        radius
    except:
        radius=500
    try:
        LIMIT
    except:
        LIMIT = 100 # A default Foursquare API limit value
        
    try:
        section
    except:
        section='trending'
    
    
    print('using-- LIMIT=', LIMIT, ', radius= ',radius, ',  section= ',section)
    
    
    
    df=pd.read_csv('SwanseaCombo.csv')
    
    ####################################
    ###############################
    
    # info from function
    swansea_venues = getNearbyVenues(names=df['GEOGRAPHYCODE'], latitudes=df['Latitude'],longitudes=df['Longitude'],radius=radius, LIMIT=LIMIT ,section=section)

    ####################################
    # swansea_venues = list of Locations with different venues
    # e.g. [0] W0001 - Art Gallery --[1] W0001 - Escape Room--[2] etc then [10] W0001 - Jazz club
    ###############################


    # one hot encoding--- i.e. create columns 1 or 0 for each category
    swansea_onehot = pd.get_dummies(swansea_venues[['Venue Category']], prefix="", prefix_sep="")#split categories into binary yes/no

    #delete check
    try:
        swansea_onehot.drop(columns='Neighborhood',inplace=True)
    except:
        pass
    
    # add neighborhood column back to dataframe
    swansea_onehot['Neighborhood'] = swansea_venues['Neighborhood'] 
       
    # move neighborhood column to the first column
    fixed_columns = [swansea_onehot.columns[-1]] + list(swansea_onehot.columns[:-1])
    swansea_onehot = swansea_onehot[fixed_columns]
    
    # group by Location ID
    swansea_grouped = swansea_onehot.groupby('Neighborhood').mean().reset_index()
    
    
    #############  SAVE
    fname=fname.split('.')[0]+'.pkl'
    with open(fname, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([swansea_grouped, swansea_venues], f)
    
    
   
    




    ############################################################################################
    
    
    ############################################################################################
    
def fit_Clusters(fname):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import copy
    import pickle
    fname=fname.split('.')[0]+'.pkl'
    # Getting back the objects:
    with open(fname,'rb') as f:  # Python 3: open(..., 'rb')
       swansea_grouped, swansea_venues = pickle.load(f)
   
    ### CLUSTER NEIGHBORHOODS
    
    # import k-means from clustering stage
    from sklearn.cluster import KMeans
    # set number of clusters
    kclusters = 10
    
    
    
    
    swansea_grouped_clustering = swansea_grouped.drop('Neighborhood', 1)
    
    # run k-means clustering
    kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(swansea_grouped_clustering)
    
    # check cluster labels generated for each row in the dataframe
    kmeans.labels_[0:10] 

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Best k
    sse=[]
    sil=[]
    maxone=[]
    maxtwo=[]
    max3=[]
    
    
    a=np.arange(2,10)
    b=np.arange(10,30,2)
    c=np.arange(30,60,6)
    d=np.arange(60,170,15)
    e=np.concatenate((a,b,c,d))
    
    for n in e:
        
        #Train Model and Predict  
        kmeans = KMeans(n_clusters=n, random_state=0).fit(swansea_grouped_clustering)
        
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(swansea_grouped_clustering)
        curr_sse = 0
        
        
        labels = kmeans.labels_
        
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(swansea_grouped_clustering)):
          curr_center = centroids[pred_clusters[i]]
          curr_sse += (swansea_grouped_clustering.iloc[i, 0] - curr_center[0]) ** 2 + (swansea_grouped_clustering.iloc[i, 1] - curr_center[1]) ** 2
            
          
        maxone.append(pd.value_counts(kmeans.labels_).max()/pd.value_counts(kmeans.labels_).sum())
        maxtwo.append(pd.value_counts(kmeans.labels_)[1:5].max()/pd.value_counts(kmeans.labels_).sum())
        max3.append(pd.value_counts(kmeans.labels_)[2:5].max()/pd.value_counts(kmeans.labels_).sum())
        sse.append(curr_sse)
        sil.append(silhouette_score(swansea_grouped_clustering,  labels,metric = 'euclidean'))
        
        
   
    plt.plot(e,sse/max(sse))
    plt.plot(e,sil/max(sil))
    plt.legend(['Within-Cluster-Sum of Squared Errors (WSS) ','Silhouette value'],loc=1)
    plt.ylabel('Normalised error')
    plt.xlabel('K')
    plt.show()
    
     # # set number of clusters
    print('Enter K value (integer)- look for Elbow   ')
    kclusters=input()
    kclusters=int(kclusters)
    print('Using k as -->',kclusters)
    
   
    
    
    ############------------------------------->
    # run k-means clustering
    kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(swansea_grouped_clustering)
    
    #------------------------------------------->


    ####play with dFs
    
    
      ##############################
    def return_most_common_venues(row, num_top_venues):
        row_categories = row.iloc[1:]
        row_categories_sorted = row_categories.sort_values(ascending=False)
        
        return row_categories_sorted.index.values[0:num_top_venues]
    
    
    num_top_venues = 10
    #
    indicators = ['st', 'nd', 'rd']
    
    # create columns according to number of top venues
    columns = ['Neighborhood']
    for ind in np.arange(num_top_venues):
        try:
            columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
        except:
            columns.append('{}th Most Common Venue'.format(ind+1))
            
    

    # create a new dataframe
    neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
    neighborhoods_venues_sorted['Neighborhood'] = swansea_grouped['Neighborhood']
    
    for ind in np.arange(swansea_grouped.shape[0]):
        neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(swansea_grouped.iloc[ind, :], num_top_venues)

    
    neighborhoods_venues_sorted['Cluster Labels']=kmeans.labels_
    
    df=pd.read_csv('SwanseaCombo.csv')
    swansea_merged=df[['GEOGRAPHYCODE','Latitude','Longitude']]
    
    ####JOIN SWANSEA MERGED (POSTAL/LONG DATA) WITH NEIGH SORTED (FREQ DATA)

    # merge SWANS_grouped with SWNAS_data to add latitude/longitude for each neighborhood
    swansea_merged_ = copy.copy(swansea_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='GEOGRAPHYCODE') ) 
    
    count_df=pd.DataFrame(swansea_merged_['Cluster Labels'].value_counts())
    
    # join the count data frame back with the original data frame
    new_index = count_df.merge(swansea_merged_[['Cluster Labels']], left_index=True, right_on='Cluster Labels')
        
    # output the original data frame in the order of the new index.
    swansea_merged_=swansea_merged_.reindex(new_index.index)
    
    ####ADD COUNTS ON END AND REORDER BASED ON THIS
    
    temp=swansea_merged_.loc[:,['GEOGRAPHYCODE','Cluster Labels']].groupby(['Cluster Labels']).count()
    # temp=pd.DataFrame(temp.index,temp.Neighborhood)
    temp.reset_index(inplace=True)
    temp.rename(columns={'GEOGRAPHYCODE':'Cluster count'},inplace=True)
    temp.sort_values('Cluster count',inplace=True)#['Cluster Labels']
    
    swansea_merged_=swansea_merged_.merge(temp,left_on='Cluster Labels',right_on='Cluster Labels')
    swansea_merged_.sort_values('Cluster count',inplace=True,ascending=False)


    #===========================================
    ### add most populat venues into swansea_merged_
    aa=[]
    bb=[]
    cc=[]
    dd=[]
    ee=[]
    ff=[]
    mostcommon=['1st Most Common Venue','2nd Most Common Venue','3rd Most Common Venue','4th Most Common Venue','5th Most Common Venue']
    import copy
    for locas in swansea_merged_.loc[:,'Cluster Labels'].unique():
        swantemp=copy.copy(swansea_merged_[swansea_merged_['Cluster Labels']==locas])
        
        swantemp_=copy.copy(swantemp.loc[:,mostcommon[0]].value_counts())
        bb.append(swantemp_.index[0])
        swantemp_=copy.copy((swantemp.loc[:,mostcommon[1]].value_counts()))
        cc.append(swantemp_.index[0])
        swantemp_=copy.copy((swantemp.loc[:,mostcommon[2]].value_counts()))
        dd.append(swantemp_.index[0])
        swantemp_=copy.copy((swantemp.loc[:,mostcommon[3]].value_counts()))
        ee.append(swantemp_.index[0])
        swantemp_=copy.copy((swantemp.loc[:,mostcommon[4]].value_counts()))
        ff.append(swantemp_.index[0])
        
        aa.append(locas)
        
    xx=pd.DataFrame(aa)
    xx=xx.rename(columns={0:'Cluster Labels'})
    xx['1st']=bb
    xx['2nd']=cc
    xx['3rd']=dd
    xx['4th']=ee
    xx['5th']=ff
    
    
    #===========================================

    ##############################
    ###Folium plot
    
    
    import matplotlib.colors as colors
    import random
    import copy
    import folium 
    
    
    swanPlot=swansea_merged_
    # create map
    map_clusters = folium.Map(location=[swanPlot['Latitude'].iloc[0], \
                                        swanPlot['Longitude'].iloc[0]], zoom_start=11, control_scale=True,tiles="Stamen Toner")
    
    # set color scheme for the clusters
    N = kclusters+5#add a bit as color returns to same
    import seaborn as sns
    colors_array = sns.color_palette('hls', N)#husl hls Paired
    rainbow = [colors.rgb2hex(i) for i in colors_array]
    
    rainbow2=copy.copy(rainbow)
    random.shuffle(rainbow)
    
    labelUnique=swansea_merged_['Cluster Labels'].unique()
    # add markers to the map
    
    i=1
    ii=0
    MAXadj=swanPlot['Cluster count'].max()
    for lat, lon, poi, cluster, cluster_count, venueNo1, venueNo2, venueNo3 in zip(swanPlot['Latitude'], swanPlot['Longitude'], \
                     swanPlot['GEOGRAPHYCODE'], swanPlot['Cluster Labels'], swanPlot['Cluster count'], \
                       swanPlot['1st Most Common Venue'],swanPlot['2nd Most Common Venue'],swanPlot['3rd Most Common Venue']):
        label = folium.Popup(str(poi) + ' Cluster ' + str(cluster) + ' ' + \
                             str(venueNo1) + ', ' + str(venueNo2) + ', ' + str(venueNo3) ,parse_html=True)
        
        try:
    
            cluster=int(cluster)
            ii=np.where(labelUnique==cluster)[0][0]
            print(ii)
            folium.CircleMarker(
            [lat, lon],
            radius=4+(2*(cluster_count/MAXadj)),
            popup=label,
            color=rainbow2[ii],
            weight =2,
            fill=True,
            fill_color=rainbow[ii],
            fill_opacity=0.7).add_to(map_clusters)
        except:
            pass
        i=i+1    
    
    print('got here') 
    
    map_clusters

    return map_clusters 
################################################################



################################################################

################################################################    


    ############################################################################################
    
    
    ############################################################################################
    
def fit_Clusters_myLoc():
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import copy
    
    
    
    #load data
    df=pd.read_csv('SwanseaCombo_WivPricesWiv4sq.csv')
    
    #some adjustments
    df=df.drop(columns='Unnamed: 0')
    df=df.set_index('GEOGRAPHYCODE')
    df=df.drop(columns='NoOfPpl')
    
    #just one value for flats
    df['PcFlats']=df[['PcFlatpurposebuilt','PcFlatofHse','PcFlatCommercial']].sum(axis=1)
    
    colchoice=1
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
    
    df=df[colchoice]
    df=df.drop(columns='Price')
    
    normQ=1#normalise the data?
    if normQ==1:
        swansea_grouped_clustering=copy.copy( (df-df.min())/(df.max()-df.min()) )
        dfmin=df.min()
        dfmax=df.max()
    #         df_*(dfmax-dfmin)+dfmin
    else:
        swansea_grouped_clustering=copy.copy( df )
    
    # import k-means from clustering stage
    from sklearn.cluster import KMeans
    # set number of clusters
    kclusters = 10
        
    # run k-means clustering
    kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(swansea_grouped_clustering)
    
    # check cluster labels generated for each row in the dataframe
    kmeans.labels_[0:10] 
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Best k
    sse=[]
    sil=[]
    maxone=[]
    maxtwo=[]
    max3=[]
    
    
    a=np.arange(2,10)
    b=np.arange(10,30,2)
    c=np.arange(30,60,6)
    d=np.arange(60,170,15)
    e=np.concatenate((a,b,c,d))
    
    for n in e:
    
        #Train Model and Predict  
        kmeans = KMeans(n_clusters=n, random_state=0).fit(swansea_grouped_clustering)
    
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(swansea_grouped_clustering)
        curr_sse = 0
    
    
        labels = kmeans.labels_
    
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(swansea_grouped_clustering)):
          curr_center = centroids[pred_clusters[i]]
          curr_sse += (swansea_grouped_clustering.iloc[i, 0] - curr_center[0]) ** 2 + (swansea_grouped_clustering.iloc[i, 1] - curr_center[1]) ** 2
    
    
        maxone.append(pd.value_counts(kmeans.labels_).max()/pd.value_counts(kmeans.labels_).sum())
        maxtwo.append(pd.value_counts(kmeans.labels_)[1:5].max()/pd.value_counts(kmeans.labels_).sum())
        max3.append(pd.value_counts(kmeans.labels_)[2:5].max()/pd.value_counts(kmeans.labels_).sum())
        sse.append(curr_sse)
        sil.append(silhouette_score(swansea_grouped_clustering,  labels,metric = 'euclidean'))
    
    
    
    plt.plot(e,sse/max(sse))
    plt.plot(e,sil/max(sil))
    plt.legend(['Within-Cluster-Sum of Squared Errors (WSS) ','Silhouette value'],loc=1)
    plt.ylabel('Normalised error')
    plt.xlabel('K')
    plt.show()
    
     # # set number of clusters
    print('Enter K value (integer)- look for Elbow   ')
    kclusters=input()
    kclusters=int(kclusters)
    print('Using k as -->',kclusters)
    

    
   
    
    
    ############KMEans------------------------------->
    # run k-means clustering
    kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(swansea_grouped_clustering)
    
    #------------------------------------------->


    swansea_grouped_clustering['Cluster Labels']=kmeans.labels_
    swansea_merged_=pd.read_csv('SwanseaCombo.csv',usecols=['GEOGRAPHYCODE','Latitude','Longitude'])
    
    
    ####JOIN SWANSEA MERGED (POSTAL/LONG DATA) WITH NEIGH SORTED (FREQ DATA)
    
    # merge SWANS_grouped with SWNAS_data to add latitude/longitude for each neighborhood
    swansea_merged_ = (swansea_merged_.join(swansea_grouped_clustering, on='GEOGRAPHYCODE') ) 
    
    count_df=pd.DataFrame(swansea_merged_['Cluster Labels'].value_counts())
    
    # join the count data frame back with the original data frame
    new_index = count_df.merge(swansea_merged_[['Cluster Labels']], left_index=True, right_on='Cluster Labels')
    
    # output the original data frame in the order of the new index.
    swansea_merged_=swansea_merged_.reindex(new_index.index)

    

    
    ####ADD COUNTS ON END AND REORDER BASED ON THIS
    
    temp=copy.copy(swansea_merged_.loc[:,['GEOGRAPHYCODE','Cluster Labels']].groupby(['Cluster Labels']).count())
    # temp=pd.DataFrame(temp.index,temp.Neighborhood)
    temp.reset_index(inplace=True)
    temp.rename(columns={'GEOGRAPHYCODE':'Cluster count'},inplace=True)
    temp.sort_values('Cluster count',inplace=True)#['Cluster Labels']
    
    swansea_merged_=swansea_merged_.merge(temp,left_on='Cluster Labels',right_on='Cluster Labels')
    swansea_merged_.sort_values('Cluster count',inplace=True,ascending=False)


    #############################
    ####Folium plot
    
    
    
    import matplotlib.colors as colors
    import random
    import copy
    import folium 
    
    
    swanPlot=swansea_merged_
    # create map
    map_clusters = folium.Map(location=[swanPlot['Latitude'].iloc[0], swanPlot['Longitude'].iloc[0]], zoom_start=11, control_scale=True,tiles="Stamen Toner")
    # #         ”OpenStreetMap”,”Stamen Terrain”, “Stamen Toner”, “Stamen Watercolor”, ”CartoDB positron”, “CartoDB dark_matter”

    
    # set color scheme for the clusters
    N = kclusters+5#color retruns
    import seaborn as sns
    colors_array = sns.color_palette('hls', N)#husl hls Paired
    rainbow = [colors.rgb2hex(i) for i in colors_array]
    
    rainbow2=copy.copy(rainbow)
    random.shuffle(rainbow)
    
    
    # add markers to the map
    labelUnique=swansea_merged_['Cluster Labels'].unique()
    i=1
    ii=0
    MAXadj=swanPlot['Cluster count'].max()
    for lat, lon, poi, cluster, cluster_count in zip(swanPlot['Latitude'], swanPlot['Longitude'], \
                     swanPlot['GEOGRAPHYCODE'], swanPlot['Cluster Labels'], swanPlot['Cluster count']):
        label = folium.Popup(str(poi) + ' Cluster ' + str(cluster) ,parse_html=True)
        
        try:
    
            cluster=int(cluster)
            ii=np.where(labelUnique==cluster)[0][0]
    
            folium.CircleMarker(
            [lat, lon],
            radius=4+(4*(cluster_count/MAXadj)),
            popup=label,
            color=rainbow2[ii],
            weight =2,
            fill=True,
            fill_color=rainbow[ii],
            fill_opacity=0.7).add_to(map_clusters)
        except:
            pass
        i=i+1    
    
    print('got here') 
    
    map_clusters

    
    
    #===========================================

    

    return map_clusters 
################################################################



################################################################

################################################################ 
    
def getNearbyVenues(names, latitudes, longitudes, radius,LIMIT,section):
    import pandas as pd
    from configparser import ConfigParser
    import requests

    parser = ConfigParser()
    _ = parser.read('nb.cfg')
    CLIENT_ID=parser.get('my_4square', 'CLIENT_ID')
    CLIENT_SECRET = parser.get('my_4square', 'CLIENT_SECRET')
    VERSION = '20180605' # Foursquare API version
    
    # LIMIT = 100 
    
    ii=0
    
    venues_list=[]
    
    for name, lat, lng in zip(names, latitudes, longitudes):
        
        print(name)

        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}&sortByPopularity=1&time=any&day=any&&section={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT,
            section)
        
        ## just a check can be long!
        if ii==0:
            print('Could take a while press any key to continue')
            input()
            
            
        try:
            # make the GET request
            results = requests.get(url).json()["response"]['groups'][0]['items']

            # return only relevant information for each nearby venue
            venues_list.append([(
                name, 
                lat, 
                lng, 
                v['venue']['name'], 
                v['venue']['location']['lat'], 
                v['venue']['location']['lng'],  
                v['venue']['categories'][0]['name']) for v in results])

            nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
            
            nearby_venues.columns = ['Neighborhood', 
                          'Neighborhood Latitude', 
                          'Neighborhood Longitude', 
                          'Venue', 
                          'Venue Latitude', 
                          'Venue Longitude', 
                          'Venue Category']
            ii=ii+1
        except:
            nearby_venues=[]
    
    return nearby_venues