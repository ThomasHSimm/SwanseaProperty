def ChoroPlot(whattopplot,fname):
    #call like this
        #import myFoliumPlot as mFP
        #m=mFP.ChoroPlot('Deprivation')
        #m
        #saving
#         m.save('RandomForestPredPricDiff.html')
        # https://pdfresizer.com/convert
    #speed up and improve!
    import pandas as pd
    import geopandas as gpd
    import folium
    

    print('here')

    ## load data
#     df_=pd.read_csv('SwanseaCombo_WivPricesWiv4sq.csv')
#     df_=df_['GEOGRAPHYCODE']
    df=pd.read_csv(fname)
    df=df.reset_index()
    df=df[['GEOGRAPHYCODE',whattopplot]]

    fname='wales_oa_2011.shp'
    nil = gpd.read_file(fname)
    nil.head()

    #convert to lat lng
    nil['geometry'] = nil['geometry'].to_crs(epsg=4326)

    #get rid of columns
    nil=nil[['code','geometry']]
    #rename code to match my data
    nil=nil.rename(columns={'code':'GEOGRAPHYCODE'})

    # merge data frames
    nilpop=nil.merge(df,on="GEOGRAPHYCODE")
        
    #a middle for map
    x_map=nilpop.centroid.x.mean()
    y_map=nilpop.centroid.y.mean()
    
    #initial map
    m = folium.Map(location=[y_map,x_map], zoom_start=10,control_scale=True,tiles="Stamen Toner")#,tiles = t_list[1])
    folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(m)

    #a scale
    myscale = (nilpop[whattopplot].quantile((0,0.02,0.1,0.25,0.5,0.75,0.9,0.98,1))).tolist()
    
    #the chloro
    folium.Choropleth(
    geo_data="swansea-wales-census.json",
    threshold_scale=myscale,
    name='choropleth',
    data= nilpop,
    columns=['GEOGRAPHYCODE',whattopplot],
    key_on= "feature.properties.GEOGRAPHYCODE",
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2
    ).add_to(m)
    folium.LayerControl().add_to(m)
    #plot the map
    


    print('and here')
    return(m)			