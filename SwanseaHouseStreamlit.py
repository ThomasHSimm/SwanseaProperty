import streamlit as st
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
    
st.title('House details in Swansea, UK')
whattoplotBIG=['Price','Predicted Price','Predicted Price Difference']
fnameBIG=['SwanseaCombo_WivPricesWiv4sq.csv','RF_PropertDetailsPred','RF_LocationPred','RF_PropertANDLocationPred','RF_LatLongPred']


    
#@st.cache
def load_dat():
    import pickle
    
    
    
    with open('nilpop_SwanseaCombo_WivPricesWiv4sq_Price.pkl','rb') as f:  # Python 3: open(..., 'rb')
       nilpop = pickle.load(f)
    nilpop['Price']=nilpop['Price']
    return nilpop

nilpop=load_dat()



def plotmap(whattopplot):
    #a middle for map
    x_map=nilpop.geometry.centroid.x.mean()
    y_map=nilpop.geometry.centroid.y.mean()

    if nilpop[whattopplot].min()<0:
            myscale=[-300 ,-200, -100, 0, 40, 80, 120]
    else:
        myscale = (nilpop[whattopplot].quantile((0,0.25,0.5,0.75,0.9,0.98,1))).tolist()
        for ii in range(np.shape(myscale)[0]):
            myscale[ii]=int(myscale[ii]-myscale[ii]%1)
        myscale[-1]=myscale[-1]+1    

    # plot map
    m = folium.Map(location=[y_map,x_map], zoom_start=10)
    
    #add choropleth
    choropleth =folium.Choropleth(
    geo_data="swansea-wales-census.json",
    threshold_scale=myscale,
    name='choropleth',
    #     label=
    legend_name=whattopplot,    
    data= nilpop,
    columns=['GEOGRAPHYCODE',whattopplot],
    key_on= "feature.properties.GEOGRAPHYCODE",
    fill_color='RdYlBu',#PuBuGn YlGn PuBuGn YlGnBu
    fill_opacity=0.7,
    line_opacity=0.2
    )
    
    #get rid of legend
    for key in choropleth._children:
        if key.startswith('color_map'):
            del(choropleth._children[key])
            
    #add choropleth
    choropleth.add_to(m)
    
    #add layer control option
    folium.LayerControl().add_to(m)

    
    #what does when mouseover
    choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(fields=['GEOGRAPHYCODE','Price'],
            aliases=['Neighborhood','Price'],sticky=True)
    )
    
# folium.features.GeoJsonTooltip(['GEOGRAPHYCODE'],labels=False)    
    
    return m, choropleth

colSel=nilpop.columns
colSel=colSel[2:]

#some side menus to slect what to plot
option = st.sidebar.selectbox(
    'What map do you want to plot?',
     colSel)

m, choropleth=plotmap(option)
# add legend
option
choropleth.color_scale
# call to render Folium map in Streamlit
folium_static(m)

#some side menus to slect what to plot
option1 = st.selectbox(
    'What do you want to plot on x?',
     colSel)

#some side menus to slect what to plot
option2 = st.selectbox(
    'What do you want to plot on y?',
     colSel)

# in case both options are the same
i=0
while option ==option2:
    option=colSel[i]
    i=i+1

#tell user what they selected
'You selected: ', option, 'and', option2

#Plot the data
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots()
plt.plot(nilpop[option],nilpop[option2],'sk',markersize=12,linewidth=3,mfc='w')
#more on the figure
plt.xlabel(option)
plt.ylabel(option2)
plt.grid()

#plot the figure on st
st.pyplot(fig)



