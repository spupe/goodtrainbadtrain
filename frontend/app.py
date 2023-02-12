### Imports and functions

from numpy import ones_like
from timeit import default_timer as timer
import streamlit as st
import pandas as pd
import datetime
import folium
import time
import requests
from datetime import datetime, time, timedelta
from streamlit_folium import folium_static
from pyhafas import HafasClient
from pyhafas.profile import VSNProfile

@st.cache
def create_journeydf(journeylist):
    # TODO: implement proper stops along the way. DB API gives back a list of stations with coordinates, ie: journeys[0].legs[0].stopovers[0].stop.latitude
    journey_start = []
    journey_end = []    
    journey_origin = []
    journey_destination = []
    journey_numberlegs = []
    leg1_train = []
    leg1_destination = []
    leg1_datetime = []
    leg2_train = []
    leg2_destination = []
    leg2_datetime = []
    leg3_train = []
    leg3_destination = []
    leg3_datetime = []
  
    for journey in journeylist:
        journey_origin.append(journey.legs[0].origin.name)
        journey_destination.append(journey.legs[-1].destination.name)
        journey_start.append(journey.legs[0].departure.strftime("%H:%M"))
        journey_end.append(journey.legs[-1].arrival.strftime("%H:%M"))
        journey_numberlegs.append(len(journey.legs))
        leg1_train.append(journey.legs[0].name)
        leg1_destination.append(journey.legs[0].destination.name)
        leg1_datetime.append(journey.legs[0].departure.strftime("%Y-%m-%d %H:%M:%S"))
        
        if len(journey.legs) == 1:
            leg2_train.append(' ')
            leg2_destination.append(' ')
            leg2_datetime.append(' ')
            leg3_train.append(' ')
            leg3_destination.append(' ')
            leg3_datetime.append(' ')
        elif len(journey.legs) == 2:
            leg2_train.append(journey.legs[1].name)
            leg2_destination.append(journey.legs[1].destination.name)
            leg2_datetime.append(journey.legs[1].departure.strftime("%Y-%m-%d %H:%M:%S"))
            leg3_train.append(' ')
            leg3_destination.append(' ')
            leg3_datetime.append(' ')
        elif len(journey.legs) == 3:
            leg2_train.append(journey.legs[1].name)
            leg2_destination.append(journey.legs[1].destination.name)
            leg2_datetime.append(journey.legs[1].departure.strftime("%Y-%m-%d %H:%M:%S"))
            leg3_train.append(journey.legs[2].name)
            leg3_destination.append(journey.legs[2].destination.name)
            leg3_datetime.append(journey.legs[2].departure.strftime("%Y-%m-%d %H:%M:%S"))
        elif len(journey.legs) == 4: #there are somehow some trips with four stops, I'm filtering those out later
            leg2_train.append(journey.legs[1].name)
            leg2_destination.append(journey.legs[1].destination.name)
            leg2_datetime.append(journey.legs[1].departure.strftime("%Y-%m-%d %H:%M:%S"))
            leg3_train.append(journey.legs[2].name)
            leg3_destination.append(journey.legs[2].destination.name)
            leg3_datetime.append(journey.legs[2].departure.strftime("%Y-%m-%d %H:%M:%S"))

    journeydf = pd.DataFrame(columns = ['Start','End','Origin','Destination','Connections',
                                         'Leg1_Dest','Leg1_Train','Leg1_Datetime',
                                         'Leg2_Dest', 'Leg2_Train','Leg2_Datetime',
                                         'Leg3_Dest','Leg3_Train', 'Leg3_Datetime'])

    journeydf.Start = journey_start
    journeydf.End = journey_end 
    journeydf.Origin = journey_origin
    journeydf.Destination = journey_destination
    journeydf.Connections = journey_numberlegs            
    journeydf.Leg1_Dest = leg1_destination
    journeydf.Leg1_Train = leg1_train
    journeydf.Leg1_Datetime = leg1_datetime
    journeydf.Leg2_Dest = leg2_destination
    journeydf.Leg2_Train = leg2_train
    journeydf.Leg2_Datetime = leg2_datetime
    journeydf.Leg3_Dest = leg3_destination
    journeydf.Leg3_Train = leg3_train
    journeydf.Leg3_Datetime = leg3_datetime
    journeydf.Origin.replace('Köln Messe/Deutz Gl.11-12', 'Köln Messe',inplace=True)
    journeydf.Destination.replace('Köln Messe/Deutz Gl.11-12', 'Köln Messe',inplace=True)
    journeydf.Destination.replace('Köln Messe/Deutz Gl.11-12', 'Köln Messe',inplace=True)
    journeydf = journeydf.sort_values(by=['Connections']) # Sort so that trains with fewer connections get listed first - we can later sort by our model predictions
    journeydf = journeydf.reset_index(drop=True)
    
    return journeydf

# @st.cache # st cache doesn't work with map object, at least so far
def create_map(leglist1,leglist2,leglist3,checkbox1,checkbox2,checkbox3):

    m = folium.Map(width=1000,height=550,location=[50.6589, 13.3383], zoom_start=6, tiles ='Cartodb dark_matter')#'cartodbpositron')#'Stamen Toner')#'openstreetmap')

    normalweight=2
    selectedweight=4
    if checkbox1:
        aline1=folium.PolyLine(locations=leglist1,weight=selectedweight,color = '#E55635',no_clip=True)
        m.add_child(aline1) 
        for i in leglist1:
            folium.CircleMarker(location=[i[0], i[1]],
                                radius=3,
                                color='#E55635',
                                fill_color='#E55635',
                                fill=True).add_to(m)
        coords = i
    else:
        aline1=folium.PolyLine(locations=leglist1,weight=normalweight,color = 'white',no_clip=True)
        #m.add_child(aline1) 

    if checkbox2:
        aline2=folium.PolyLine(locations=leglist2,weight=selectedweight,color = '#E55635',no_clip=True)
        m.add_child(aline2)
        for i in leglist2:
            folium.CircleMarker(location=[i[0], i[1]],
                                radius=3,
                                color='#E55635',
                                fill_color='#E55635',
                                fill=True).add_to(m)
    else:
        aline2=folium.PolyLine(locations=leglist2,weight=normalweight,color = 'white',no_clip=True)
        #m.add_child(aline2) 

    if checkbox3:
        aline3=folium.PolyLine(locations=leglist3,weight=selectedweight,color = '#E55635',no_clip=True)
        m.add_child(aline3)
        for i in leglist3:
            folium.CircleMarker(location=[i[0], i[1]],
                                radius=3,
                                color='#E55635',
                                fill_color='#E55635',
                                fill=True).add_to(m)
    else:
        aline3=folium.PolyLine(locations=leglist3,weight=normalweight,color = 'white',no_clip=True)
        #m.add_child(aline3) 

    return m

# placeholders

def calldb_preprocess(origin,destination,date):
    origin1 = client.locations(origin)[0]
    destination1 = client.locations(destination)[0]
    print("debug_preprocess",origin1,destination1,date)
    journeys = client.journeys(
    origin=origin1,
    destination=destination1,
    date=date,
    max_changes=2,
    min_change_time=15,
    max_journeys=-1,
    products={
        'long_distance_express': True,
        'regional_express': False,
        'regional': False,
        'suburban': False,
        'bus': False,
        'ferry': False,
        'subway': False,
        'tram': False,
        'taxi': False
    })



    journeydf = create_journeydf(journeys)

    leglist = [journeydf.Origin,journeydf.Leg1_Dest,journeydf.Leg2_Dest,journeydf.Leg3_Dest]
    leglist1 = [leglist[0][0],leglist[1][0],leglist[2][0],leglist[3][0],' '] # We are adding ' ' for the trick below
    leglist2 = [leglist[0][1],leglist[1][1],leglist[2][1],leglist[3][1],' ']
    leglist3 = [leglist[0][2],leglist[1][2],leglist[2][2],leglist[3][2],' ']
    leglist1 = leglist1[0:leglist1.index(' ')] # Find out where ' ' first appears in the list, and the position before that is the end destination, and you can work from there. Useful for coordinates
    leglist2 = leglist2[0:leglist2.index(' ')]
    leglist3 = leglist3[0:leglist3.index(' ')]
    print(leglist1,leglist2,leglist3)
    my_dict = {'Köln Hbf': [50.9432, 6.9586], 
                'Köln Messe': [50.9469, 6.9832], 'Köln Messe/Deutz Gl.11-12': [50.9469, 6.9832],
                'München Hbf': [48.1403, 11.5600], 'Mannheim Hbf': [49.4796, 8.4699],
                'Frankfurt(Main) Flughafen Fernbf': [50.0529, 8.5698], 'Frankfurt(Main) Hbf': [50.0529, 8.5698], 'Frankfurt(Main) Süd': [50.0529, 8.5698],
                'Nürnberg Hbf': [49.4456, 11.0825], 'Würzburg Hbf': [49.8024, 9.9356], 
                'Berlin Hbf (tief)': [52.5251, 13.4694] ,'Berlin Hbf': [52.5251, 13.3694], 'Berlin Spandau': [52.5251, 13.3694], 
                'Hamburg Hbf': [53.5530, 10.0066],  'Hamm(Westf)':  [851.6784, 7.8089],
                'Düsseldorf Hbf': [51.2198, 6.7945], 'Stuttgart Hbf':[48.7832, 9.1823], 
                'Essen Hbf': [51.4504, 7.0129] , 'Hagen(Westf) Bahnhof': [51.3626, 7.4617],
                'Hagen Hbf':  [51.3626, 7.4617], 'Hannover Hbf': [52.3765, 9.7410],
                'Erfurt Hbf': [50.9725, 11.0380],'Göttingen': [51.5366, 9.9268],
                'Solingen Hbf': [51.1652,7.0671]}
    leglist1[:]=[my_dict.get(e,'') for e in leglist1]
    leglist2[:]=[my_dict.get(e,'') for e in leglist2]
    leglist3[:]=[my_dict.get(e,'') for e in leglist3]
    print(leglist1)
    print(leglist2)
    print(leglist3)

    return journeydf,leglist1,leglist2,leglist3


def callapi(pickup,dropoff,querydate='2023-10-02 15:15:00',train='ICE 109'):
    print("debug2",pickup,dropoff,querydate,train)
    try:
        response = requests.get('https://finalapp22june-4muwooak2q-ew.a.run.app/predict',
        params={'start_city': pickup, 'end_city': dropoff, 'user_date': querydate, 'ice': train})
        print(response)
        response = response.json()
        start_city = response["start_city"]
        end_city = response["end_city"]
        train = response["train"]
        prediction = response["prediction"]
        probability = response["probability"]
        weather = response["expected_weather_conditions"]
        mean_delay = response["mean_delay"]
        print([start_city,end_city,train,prediction,probability,weather,mean_delay])
        return [start_city,end_city,train,prediction,probability,weather,mean_delay]
    except:
        st.warning('API call failed. No prediction able to be produced at this time. Please change parameters and try again.')
        print(response)


### Main
#print('MAIN')
if 'active' not in st.session_state:
    st.session_state['active'] = False
if 'maindf' not in st.session_state:
	st.session_state['maindf'] = []
if 'checkbox1' not in st.session_state:
	st.session_state.checkbox1 = True
if 'checkbox2' not in st.session_state:
	st.session_state.checkbox2 = False
if 'checkbox3' not in st.session_state:
	st.session_state.checkbox3 = False
if 'goodtrain1' not in st.session_state:
	st.session_state.goodtrain1 = True
if 'goodtrain2' not in st.session_state:
	st.session_state.goodtrain2 = True
if 'goodtrain3' not in st.session_state:
	st.session_state.goodtrain3 = False
if 'delay1' not in st.session_state:
	st.session_state.delay1 = 0
if 'delay2' not in st.session_state:
	st.session_state.delay2 = 0
if 'delay3' not in st.session_state:
	st.session_state.delay3 = 0
if 'weather1' not in st.session_state:       
    st.session_state.weather1 = 1
if 'weather2' not in st.session_state:       
    st.session_state.weather2 = 1
if 'weather3' not in st.session_state:       
    st.session_state.weather2 = 1
if 'proba1' not in st.session_state:       
    st.session_state.weather1 = 1
if 'proba2' not in st.session_state:       
    st.session_state.weather2 = 1
if 'proba3' not in st.session_state:       
    st.session_state.weather2 = 1
if 'leglist1' not in st.session_state:
	st.session_state['leglist1'] = []
if 'leglist2' not in st.session_state:
	st.session_state['leglist2'] = []
if 'leglist3' not in st.session_state:
	st.session_state['leglist3'] = []

checkbox1 = st.session_state.checkbox1
checkbox2 = st.session_state.checkbox2
checkbox3 = st.session_state.checkbox3

st.set_page_config(page_title='Good Train Bad Train', page_icon='frontend/gtbt_good.png', layout="centered", initial_sidebar_state="auto", menu_items=None)
st.image('frontend/gtbtlogo_black.png')
client = HafasClient(VSNProfile())

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(229, 86, 53);
}
</style>""", unsafe_allow_html=True)

tomorrowdate = datetime.now() + timedelta(days=1)
maxdate = datetime.now() + timedelta(days=15)
datetodisplay = tomorrowdate.date()

col0, col1,col3,col4,col5 = st.columns([4,4,3,3,2])
origin = 'Köln Hbf'
start_city = 'Koln'
destination = 'München Hbf'
end_city = 'Munchen'
with col0:
    pickup = st.selectbox(
        'From',
        ('Köln', 'Berlin', 'München'))
with col1:
    dropoff = st.selectbox(
        'To',
        ('München', 'Berlin', 'Köln'))
with col3:
    pickupdate = st.date_input('Date', value=datetodisplay,min_value=datetime.now(),max_value = maxdate)
with col4:
    pickuptime = st.slider(
        "Time",
        value=(time(8, 00)))
with col5:
    st.image('frontend/blackpadding.png',width=45)
    center_button = st.button('Go')
#with col6:
#    st.image('frontend/blackpadding.png',width=10)

if pickup == 'München':
    origin = 'München Hbf'
    start_city = 'Munchen'
if dropoff == 'Munchen':
    destination = 'München Hbf'
    end_city = 'Munchen'
if pickup == 'Köln':
    origin = 'Köln Hbf'
    start_city = 'Koln'
if dropoff == 'Köln':
    destination = 'Köln Hbf'
    end_city = 'Koln'
if pickup == 'Berlin':
    origin = 'Berlin Hauptbahnhof'
    start_city = 'Berlin'
if dropoff == 'Berlin':
    destination = 'Berlin Hauptbahnhof'
    end_city = 'Berlin'

pickuptime = pickuptime.strftime("%H:%M")
pickuptime = pickuptime + ':00'
pickupdate = pickupdate.strftime("%Y/%m/%d")
pickupdate = pickupdate.replace("/","-")
querydate = pickupdate + ' ' + pickuptime
datefordb = datetime.strptime(querydate, '%Y-%m-%d %H:%M:%S')


if center_button:
    start = timer()
    with st.spinner('Gathering journey information from Deutsche Bahn'):
        print("debug1",origin,destination,datefordb)
        
    journeydf,leglist1,leglist2,leglist3 = calldb_preprocess(origin,destination,datefordb)
    print("debugreply",journeydf,leglist1,leglist2,leglist3)
    if st.session_state.active == False:
       st.success('DB data collected.')
    end = timer()
    print(end - start)

    start_city = str(start_city)
    end_city = str(end_city)
 
    start = timer()
    # The API currently takes a series of predictions, as long as we divide by ',' in a string. So let's build a string in that format to query the API
    end_city1 = end_city
    end_city2 = end_city
    end_city3 = end_city
    # Note that we are only calling for the last leg for now, for demonstration purposes, as that speeds up our API call immensely
    if journeydf.Connections[0] == 1:
        start_city1 = start_city
        train1 = journeydf.Leg1_Train[0]
        querydate1 = journeydf.Leg1_Datetime[0]
    elif journeydf.Connections[0] == 2:
        if journeydf.Leg1_Dest[0] == 'Köln Messe/Deutz Gl.11-12':
            start_city1 = start_city
        else:
            start_city1 = journeydf.Leg1_Dest[0]
        train1 = journeydf.Leg2_Train[0]
        querydate1 = journeydf.Leg2_Datetime[0]        
    elif journeydf.Connections[0] == 3:
        if journeydf.Leg2_Dest[0] == 'Köln Messe/Deutz Gl.11-12':
            start_city1 = start_city
        else:
            start_city1 = journeydf.Leg2_Dest[0]

        train1 = journeydf.Leg3_Train[0]
        querydate1 = journeydf.Leg3_Datetime[0]
    
    if journeydf.Connections[1] == 1:
        start_city2 = start_city
        train2 = journeydf.Leg1_Train[1]
        querydate2 = journeydf.Leg1_Datetime[1]
    elif journeydf.Connections[1] == 2:
        if journeydf.Leg1_Dest[1] == 'Köln Messe/Deutz Gl.11-12':
            start_city2 = start_city
        else:
            start_city2 = journeydf.Leg1_Dest[1]
        train2 = journeydf.Leg2_Train[1]
        querydate2 = journeydf.Leg1_Datetime[1]
    elif journeydf.Connections[1] == 3:
        if journeydf.Leg2_Dest[1] == 'Köln Messe/Deutz Gl.11-12':
            start_city2 = start_city
        else:
            start_city2 = journeydf.Leg2_Dest[1]
        train2 = journeydf.Leg3_Train[1]
        querydate2 = journeydf.Leg1_Datetime[1]

    if journeydf.Connections[2] == 1:
        start_city3 = start_city
        train3 = journeydf.Leg1_Train[2]
        querydate3 = journeydf.Leg1_Datetime[2]
    elif journeydf.Connections[2] == 2:
        if journeydf.Leg1_Dest[2] == 'Köln Messe/Deutz Gl.11-12':
            start_city3 = start_city
        else:
            start_city3 = journeydf.Leg1_Dest[2]
        train3 = journeydf.Leg2_Train[2]
        querydate3 = journeydf.Leg1_Datetime[2]
    elif journeydf.Connections[2] == 3:
        if journeydf.Leg2_Dest[2] == 'Köln Messe/Deutz Gl.11-12':
            start_city3 = start_city
        else:
            start_city3 = journeydf.Leg2_Dest[2]
        train3 = journeydf.Leg3_Train[2]
        querydate3 = journeydf.Leg1_Datetime[2]

    # Need some input sanitization, the API only takes the name of the city, not the station. Also remove umlaut

    my_dict_cities = {'Köln Hbf': 'Koln', 'Koln': 'Koln', 'Munchen': 'Munchen', 'Frankfurt': 'Frankfurt', 
                'Berlin': 'Berlin', 'Dusseldorf': 'Dusseldorf', 'Frankfurt(Main) Süd': 'Frankfurt',
                'Köln Messe': 'Koln', 'Köln Messe/Deutz Gl.11-12': 'Koln', # Messe should never appear here but I just never want to deal with this station ever again
                'München Hbf': 'Munchen', 'Mannheim Hbf': 'Mannheim',
                'Frankfurt(Main) Flughafen Fernbf': 'Frankfurt', 'Frankfurt(Main) Hbf': 'Frankfurt',
                'Nürnberg Hbf': 'Nurnberg', 'Würzburg Hbf': 'Wurzburg', 
                'Berlin Hbf (tief)': 'Berlin' ,'Berlin Hbf': 'Berlin', 'Berlin Spandau': 'Berlin', 
                'Hamburg Hbf': 'Hamburg',  'Hamm(Westf)':  'Hamm',
                'Düsseldorf Hbf': 'Dusseldorf', 'Stuttgart Hbf': 'Stuttgart', 
                'Essen Hbf': 'Essen', 'Hagen(Westf) Bahnhof': 'Hagen',
                'Hagen Hbf':  'Hagen', 'Hannover Hbf': 'Hanover',
                'Erfurt Hbf': 'Erfurt','Göttingen': 'Gottingen',
                'Solingen Hbf': 'Solingen'}
    
    start_city1 = my_dict_cities[start_city1]
    start_city2 = my_dict_cities[start_city2]
    start_city3 = my_dict_cities[start_city3]
    end_city1 = my_dict_cities[end_city1]
    end_city2 = my_dict_cities[end_city2]
    end_city3 = my_dict_cities[end_city3]

    start_city = start_city1 + ',' + start_city2 + ','  + start_city3
    end_city = end_city1 + ',' + end_city2 + ','  + end_city3
    querydate = querydate1 + ',' + querydate2 + ','  + querydate3
    train = str(train1) + ',' + str(train2) + ','  + str(train3)
    with st.spinner('Calling our server, this could take a while'):
        try:
            apireply = callapi(pickup=start_city, dropoff=end_city,querydate=querydate,train=train)
            if st.session_state.active == False:
                st.success('Server reply gathered.')
            end = timer()
            print(end - start) 
            # start = timer()
            # with st.spinner('Now gathering info about the second journey'):
            #     train = journeydf.Leg1_Train[1]
            #     try:
            #         apireply2 = callapi(pickup=start_city, dropoff=end_city,querydate=querydate,train=train)
            #     except:
            #         train='ICE 109'
            #         apireply2 = callapi(pickup=start_city, dropoff=end_city,querydate=querydate,train=train)
            #         st.session_state.goodtrain2 = 0
            # if st.session_state.active == False:
            #     st.success('Second journey completed.')
            # end = timer()
            # print(end - start) 
            # start = timer()
            # with st.spinner('Almost there now, grabbing information for our third journey'):
            #     train=journeydf.Leg1_Train[2]
            #     try:
            #         apireply3 = callapi(pickup=start_city, dropoff=end_city,querydate=querydate,train=train)
            #     except:
            #         train='ICE 109'
            #         apireply3 = callapi(pickup=start_city, dropoff=end_city,querydate=querydate,train=train)
            #         st.session_state.goodtrain3 = 0
            # if st.session_state.active == False:
            #     st.success('Success! All done')
            # end = timer()
            # print(end - start) 

            print(journeydf)
            print(leglist1,leglist2,leglist3)
            st.session_state.active = True
            st.session_state.maindf = journeydf
            st.session_state.leglist1 = leglist1
            st.session_state.leglist2 = leglist2
            st.session_state.leglist3 = leglist3
            st.session_state.goodtrain1 = int(apireply[3].split(',')[0])
            st.session_state.goodtrain2 = int(apireply[3].split(',')[1])
            st.session_state.goodtrain3 = int(apireply[3].split(',')[2])
            print(st.session_state.goodtrain1,st.session_state.goodtrain2,st.session_state.goodtrain3)
            st.session_state.weather1 = int(apireply[-2].split(',')[0])
            st.session_state.weather2 = int(apireply[-2].split(',')[1])
            st.session_state.weather3 = int(apireply[-2].split(',')[2])
            st.session_state.delay1 = float(apireply[-1].split(',')[0])
            st.session_state.delay2 = float(apireply[-1].split(',')[1])
            st.session_state.delay3 = float(apireply[-1].split(',')[2])
            st.session_state.proba1 = float(apireply[4].split(',')[0])
            st.session_state.proba2 = float(apireply[4].split(',')[1])
            st.session_state.proba3 = float(apireply[4].split(',')[2])
            st.session_state.checkbox1 = True
            #st.experimental_rerun()           
            
        except:
            #train='ICE 109,ICE 109,ICE 109' # have to hardcode this for now, as passing a train that doesnt exist crashes our API
            #apireply = callapi(pickup=start_city, dropoff=end_city,querydate=querydate,train=train)
            #st.session_state.goodtrain1 = 0 # if train is missing in database, let's assume it's not a good 
            st.warning('API call failed. No prediction able to be produced at this time. Please change parameters and try again.')
            end = timer()
            print(end - start) 
            st.experimental_rerun()  


if st.session_state.active == True:
    journeydf = st.session_state.maindf
    leglist1 = st.session_state.leglist1
    leglist2 = st.session_state.leglist2
    leglist3 = st.session_state.leglist3
    if 'Köln Messe/Deutz Gl.11-12' in leglist1: # have to account for the cases where a "connection" is merely between two stations in Cologne
        string1 = f'{journeydf.Origin[0]} ({journeydf.Start[0]}) to {journeydf.Destination[0]} ({journeydf.End[0]}), with {journeydf.Connections[0]-2} connections'
    else:
        string1 = f'{journeydf.Origin[0]} ({journeydf.Start[0]}) to {journeydf.Destination[0]} ({journeydf.End[0]}), with {journeydf.Connections[0]-1} connections'
    if 'Köln Messe/Deutz Gl.11-12' in leglist2:
        string2 = f'{journeydf.Origin[1]} ({journeydf.Start[1]}) to {journeydf.Destination[1]} ({journeydf.End[1]}), with {journeydf.Connections[1]-2} connections'
    else:
        string2 = f'{journeydf.Origin[1]} ({journeydf.Start[1]}) to {journeydf.Destination[1]} ({journeydf.End[1]}), with {journeydf.Connections[1]-1} connections'
    if 'Köln Messe/Deutz Gl.11-12' in leglist2:
        string3 = f'{journeydf.Origin[2]} ({journeydf.Start[2]}) to {journeydf.Destination[2]} ({journeydf.End[2]}), with {journeydf.Connections[2]-2} connections'
    else:
        string3 = f'{journeydf.Origin[2]} ({journeydf.Start[2]}) to {journeydf.Destination[2]} ({journeydf.End[2]}), with {journeydf.Connections[2]-1} connections'
    
    colcheck11,colcheck12,colcheck13 =st.columns([10,3,1])
    with colcheck11:
        radiochoice=st.radio(' ',(string1,string2,string3)) 
    with colcheck12:
        st.image('frontend/blackpadding.png',width=25)
        #st.markdown(f'Confidence: {round(st.session_state.proba1,2)}%')
        #st.markdown(f'Confidence: {round(st.session_state.proba2,2)}%')
        #st.markdown(f'Confidence: {round(st.session_state.proba3,2)}%')
    with colcheck13:
        st.image('frontend/blackpadding.png',width=40)
        if st.session_state.goodtrain1 == 1:
            st.image('frontend/gtbt_good_green2.png',width=13,)
        else:
            st.image('frontend/gtbt_bad_red2.png',width=13)
        if st.session_state.goodtrain2 == 1:
            st.image('frontend/gtbt_good_green2.png',width=13,)
        else:
            st.image('frontend/gtbt_bad_red2.png',width=13)
        if st.session_state.goodtrain3== 1:
            st.image('frontend/gtbt_good_green2.png',width=13,)
        else:
            st.image('frontend/gtbt_bad_red2.png',width=13)        

    if radiochoice == string1:
        st.session_state.checkbox1=True
        st.session_state.checkbox2=False
        st.session_state.checkbox3=False
    if radiochoice == string2:
        st.session_state.checkbox1=False
        st.session_state.checkbox2=True
        st.session_state.checkbox3=False
    if radiochoice == string3:
        st.session_state.checkbox1=False
        st.session_state.checkbox2=False
        st.session_state.checkbox3=True
        
    iconcolpad0,iconcol0,iconcol1, iconcol2, iconcol3 = st.columns([1,3,3,3,3])
    with iconcolpad0:
        st.image('frontend/blackpadding.png',width=30)
    with iconcol0:
        if radiochoice == string1:
            goodtrain =  st.session_state.goodtrain1
        elif radiochoice == string2:
            goodtrain = st.session_state.goodtrain2
        elif radiochoice == string3:
            goodtrain = st.session_state.goodtrain3
        if goodtrain == 1:
            st.markdown('Our prediction')
            st.image('frontend/gtbt_good_green2.png',width=35)
        else:
            st.markdown('Our prediction')
            st.image('frontend/gtbt_bad_red2.png',width=35)
    with iconcol1:
        if radiochoice == string1:
            proba =  round(st.session_state.proba1*100)
        elif radiochoice == string2:
            proba = round(st.session_state.proba2*100)
        elif radiochoice == string3:
            proba = round(st.session_state.proba3*100)
        st.metric('Confidence',f'{proba}%')
        # if proba > 0.6:
        #     st.markdown('High confidence')
        #     css_example = '''                                                                                                                                           
        #     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
                                                                                                                                                                                                                    
        #     <i class="fa-solid fa-circle-check fa-3x" style="color:#4CAF50"></i>                                                                                                                                                                  
        #     '''
        #     st.write(css_example, unsafe_allow_html=True)    
        # else:
        #     st.markdown('Low confidence')
        #     css_example = '''                                                                                                                                           
        #     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
                                                                                                                                                                                                            
        #     <i class="fa-solid fa-circle-xmark fa-3x" style="color:#FF2C2C"></i>   
        #     ''' 
        #     st.write(css_example, unsafe_allow_html=True) 
    
    with iconcol2:
        if radiochoice == string1:
            weather =  st.session_state.weather1
        elif radiochoice == string2:
            weather = st.session_state.weather2
        elif radiochoice == string3:
            weather = st.session_state.weather3
        if weather < 3:
            st.markdown('Good weather')
            css_example = '''                                                                                                                                           
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
                                                                                                                                                                                                                    
            <i class="fa-solid fa-sun fa-2x" style="color:#4CAF50"></i>                                                                                                                                                                  
            '''
            st.write(css_example, unsafe_allow_html=True)    
        else:
            st.markdown('Bad weather')
            css_example = '''                                                                                                                                           
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
                                                                                                                                                                                                            
            <i class="fa-solid fa-poo-storm fa-2x" style="color:#FF2C2C"></i>   
            ''' 
            st.write(css_example, unsafe_allow_html=True)

    with iconcol3:
        if radiochoice == string1:
            delay =  st.session_state.delay1
        elif radiochoice == string2:
            delay = st.session_state.delay2
        elif radiochoice == string3:
            delay = st.session_state.delay3
        if delay < 10:
            st.markdown('Usually on time')    
            css_example = '''                                                                                                                                           
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
                                                                                                                                                                                                                    
            <i class="fa-solid fa-clock fa-2x" style="color:#4CAF50"></i>                                                                                                                                                                  
            '''
            st.write(css_example, unsafe_allow_html=True) 
        else:
            st.markdown('Usually delayed')    
            css_example = '''                                                                                                                                           
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
                                                                                                                                                                                                                    
            <i class="fa-solid fa-clock fa-2x" style="color:#FF2C2C"></i>                                                                                                                                                                  
            '''
            st.write(css_example, unsafe_allow_html=True)  


    ### MAP
    m = create_map(leglist1=st.session_state.leglist1,leglist2=st.session_state.leglist2,leglist3=st.session_state.leglist3,checkbox1=st.session_state.checkbox1,checkbox2=st.session_state.checkbox2,checkbox3=st.session_state.checkbox3)
    folium_static(m)  
        


    ### FOOTER

    st.markdown("""
        <style>
        .big-font {
        font-size:18px !important;
        text-align:center;
        color:white;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        .small-font {
        font-size:12px !important;
        text-align:center;
        color:white;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Developed with &#9829 by Stefano Pupe, Marie Macnee, Boris Bohsem and Juan Cotrino.<br><br></p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Hat-tip to Jose Aponte for the logo, and Johannes Schubert (zugfinder.net) for the dataset.<br><br><br></p>', unsafe_allow_html=True)


    '''
     
    
    
    
    '''
    with st.expander('Details'):
         st.markdown("<p style='text-align: center; color: white;'>We gathered records of all high-speed train rides between Cologne, Munich and Berlin in 2020 and 2021 and combined it with historical weather information for each ride. Then, we encoded real-world information in a way that could be understood by a machine learning model - for example, we filtered periods with lockdowns, flagged all major holidays and added variables indicating the week day of a given ride, as all these factors could be relevant for the model's performance. Our tests suggest that the model can predict - with about 80% accuracy - whether a given ride between these three cities will be delayed or not. All that you see here, including training and deploying the model, was built in only two weeks by four developers using open-source tools.<br><br>What is happening behind the scenes here is that we are sending the information input by the user to an API from Deutsche Bahn, which gives us a list of possible journeys on the specified date and time. Then, we send this info to our own API, which gathers weather predictions and combines that with other info we have about your ride, producing a prediction. Due to its dependence on weather, we restrict searches to a period of 15 days from the present.</h1>", unsafe_allow_html=True)

else:
    st.markdown("""
        <style>
        .big-font {
        font-size:18px !important;
        text-align:center;
        color:white;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        .small-font {
        font-size:12px !important;
        text-align:center;
        color:white;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Developed with &#9829 by Stefano Pupe, Marie Macnee, Boris Bohsem and Juan Cotrino.<br><br></p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Hat-tip to Jose Aponte for the logo, and Johannes Schubert (zugfinder.net) for the dataset.<br><br><br></p>', unsafe_allow_html=True)


    '''
    
    
    
    '''
    with st.expander('Details'):
         st.markdown("<p style='text-align: center; color: white;'>We gathered records of all high-speed train rides between Cologne, Munich and Berlin in 2020 and 2021 and combined it with historical weather information for each ride. Then, we encoded real-world information in a way that could be understood by a machine learning model - for example, we filtered periods with lockdowns, flagged all major holidays and added variables indicating the week day of a given ride, as all these factors could be relevant for the model's performance. Our tests suggest that the model can predict - with about 80% accuracy - whether a given ride between these three cities will be delayed or not. All that you see here, including training and deploying the model, was built in only two weeks by four developers using open-source tools.<br><br>What is happening behind the scenes here is that we are sending the information input by the user to an API from Deutsche Bahn, which gives us a list of possible journeys on the specified date and time. Then, we send this info to our own API, which gathers weather predictions and combines that with other info we have about your ride, producing a prediction. Due to its dependence on weather, we restrict searches to a period of 15 days from the present.</h1>", unsafe_allow_html=True)
