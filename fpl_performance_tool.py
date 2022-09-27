#%%
import json
from matplotlib.axis import YAxis
import requests
import pandas as pd
# import numpy as np
from pandas.io.json import json_normalize
import urllib.request
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# import seaborn as sns
# import unicodedata
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from bs4 import BeautifulSoup
# %%

st.set_page_config(
    page_title='FPL 22/23 Team Optimizer',
    layout='wide',
    page_icon='⚽'
)

st.header('FPL 22/23 Team Optimizer')
st.subheader('Overall Performance')

#Ask for Team ID
#%%
team_id = st.number_input(
    'Input Your Team ID:',
    step=1,
    value=4625089
)

# team_id2 = st.number_input(
#     "Input Your Competitor's ID:",
#     step=1,
#     value=1212633
# )

#With Inputted Team ID, get Leagues
str = 'https://fantasy.premierleague.com/api/entry/{}/'
urlleagueids = str.format(team_id)

with urllib.request.urlopen(urlleagueids) as f3:
    d3 = json.load(f3)

df_leauge = pd.json_normalize(d3['leagues'])

#%%
#Allow choosing of league
leaguenamelist = []

for i in list(range(0,len(df_leauge['classic'][0]))):
    df = pd.DataFrame.from_dict(df_leauge['classic'][0][i],orient='index')
    df = df.T
    leaguenamelist.append(df)

df_player_leagues = pd.concat(leaguenamelist, ignore_index=True)
options=list(df_player_leagues['name'].unique())[4:]
# options.remove('Starhub League')

league_id = st.multiselect(
    "Select League",
    options=options,
    default=options[0]
)


#Get User's Data
#%%
str = 'https://fantasy.premierleague.com/api/entry/{}/history/'
urlfpl=str.format(team_id)
# urlfpl2=str.format(team_id2)

with urllib.request.urlopen(urlfpl) as f:
    d = json.load(f)

df_fpl = pd.json_normalize(d['current'])
# df_fpl['id'] = f'{team_id}'

#Get Player IDs for all the leagues the user is in
#%%
str = 'https://fantasy.premierleague.com/api/leagues-classic/{}/standings'
# temp = 'https://fantasy.premierleague.com/api/leagues-classic/129713/standings/'
x = df_player_leagues[4:][df_player_leagues[4:]['name'] == league_id[0]]['id'].iloc[0]
urlleague = str.format(x)

with urllib.request.urlopen(urlleague) as f2:
    d2 = json.load(f2)

df_league_standings= pd.json_normalize(d2['standings'])

players_in_league_ids = []
for i in list(range(0,len(df_league_standings['results'][0]))):
    df = pd.DataFrame.from_dict(df_league_standings['results'][0][i],orient='index')
    df = df.T
    players_in_league_ids.append(df)

df_competitors = pd.concat(players_in_league_ids, ignore_index=True)

str = 'https://fantasy.premierleague.com/api/entry/{}/history/'

df_fpl2_list = []

#Get all competitor player data
for i in list(df_competitors['entry']):
    d = df_competitors[df_competitors['entry'] == i]
    url = str.format(i)
    # print(url)
    with urllib.request.urlopen(url) as f4:
        d4 = json.load(f4)
        d4 = pd.json_normalize(d4['current'])
        d4['id'] = i
        d4['Player'] = d['player_name'].iloc[0]
        df_fpl2_list.append(d4)

df_fpl2 = pd.concat(df_fpl2_list, ignore_index=True)

#Get player rank data
df_player_rank = df_fpl2[df_fpl2['id'] == team_id]
player_name = df_player_rank['Player'].loc[df_player_rank.index[0]]



# str = 'https://fantasy.premierleague.com/api/entry/{}/'
# urlleagueids = str.format(team_id)

# with urllib.request.urlopen(urlleagueids) as f3:
#     d3 = json.load(f3)

# df_leauge = pd.json_normalize(d3['leagues'])

# leaguelist = []

# #%%
# for i in list(range(0,len(df_leauge['classic'][0]))):
#     id = df_leauge['classic'][0][i]['id']
#     leaguelist.append(id)

# playerleagues = leaguelist[4:]

# str = 'https://fantasy.premierleague.com/api/leagues-classic/{}/standings'





#Insert Key Metrics, Graph of Points & Rank Animation
# %%
col1,col2,col3,col4,col5 = st.columns(5)

with col1:
    a = df_fpl['points'].loc[df_fpl.index[-1]]
    b = df_fpl['points'].loc[df_fpl.index[-1]] - df_fpl['points'].loc[df_fpl.index[-2]]
    gw = df_fpl['event'].loc[df_fpl.index[-1]]

    st.metric(
        label= f'Gameweek {gw} Points',
        value=a,
        delta= int(b)
    )

with col2:
    a1 = df_fpl['total_points'].loc[df_fpl.index[-1]]
    b1 = df_fpl['total_points'].loc[df_fpl.index[-1]] - df_fpl['total_points'].loc[df_fpl.index[-2]]
    gw1 = df_fpl['event'].loc[df_fpl.index[-1]]

    st.metric(
        label= 'Total Points',
        value=a1,
        delta= int(b1)
    )  

with col3:
    a2 = df_fpl['total_points'].loc[df_fpl.index[-1]]/len(df_fpl.index)
    b2 = df_fpl['total_points'].loc[df_fpl.index[-2]]/len(df_fpl.index[0:-1])
    c2 = a2-b2

    st.metric(
        label= 'Average Points Per GW',
        value=round(a2,2),
        delta= round(c2,2)
    )

with col4:
    a3 = sum(df_fpl['points'].tail(5))/5
    b3 = sum(df_fpl['points'][df_fpl.index[-6]:df_fpl.index[-1]])/5
    c3 = a3-b3

    st.metric(
        label= 'Average Points Past 5 GW',
        value=round(a3,2),
        delta= round(c3,2)
    )

with col5:
    a4=df_fpl['value'].loc[df_fpl.index[-1]]/10
    b4=df_fpl['value'].loc[df_fpl.index[-2]]/10
    c4=a4-b4

    st.metric(
        label='Team Value (Millions)',
        value=a4,
        delta=int(c4)
    )

# df_fpl['id'] = f'{team_id}'
# df_fpl2['id'] = f'{team_id2}'


#Insert W-o-W Points and Rank
# %%
fig = px.scatter(df_fpl2, x='event',y='total_points',color='Player',animation_frame='event',range_x=[0,38],range_y=[0,2800],size='points', size_max=15)
# st.plotly_chart(fig,use_container_width=True)
# %%

fig2 = px.bar(df_fpl2, x='total_points',y='Player',color='Player',orientation='h',animation_frame='event',range_x=[0,2700])
fig2.update_layout(yaxis={'categoryorder':'total ascending'},legend=dict(orientation='h'), margin=dict(l=10, r=10, t=10, b=10),modebar_remove=['zoom', 'select'],dragmode = False)
# st.plotly_chart(fig2,use_container_width=True)
figa1,figa2 = st.columns(2)
with figa1:
    st.text('  ')
    st.subheader('W-o-W Rankings Race')
    st.text('  ')
    st.text('  ')
    st.plotly_chart(fig2,use_container_width=True)
with figa2:
    # df_fpl_2_2 = df_fpl2[['Player']].convert_dtypes()
    player = st.multiselect('Rank Progress: Select Players To Compare With',options=df_fpl2['Player'].unique(),default=player_name)
    df_rank_graph = df_fpl2.query("Player == @player")
    fig3 = px.line(df_rank_graph, x='event', y='overall_rank', color='Player', markers=True)
    fig3.update_yaxes(autorange="reversed")
    fig3.update_layout(legend=dict(orientation='h'), margin=dict(l=10, r=10, t=10, b=10),modebar_remove=['zoom', 'select'],dragmode = False)
    st.plotly_chart(fig3,use_container_width=True)

#%%
st.subheader('W-o-W Player Team Performance')

# gameweek = st.multiselect(
#     'Select Gameweek',
#     options=range(1,39),
#     default=list(df_player_rank['event'].tail(1))[0]
# )

gameweek = st.number_input(
    'Input Gameweek',
    min_value=1,
    max_value=38,
    value=list(df_player_rank['event'].tail(1))[0]
)

gameweek2 = gameweek - 1
# gameweek = gameweek[0]

str = 'https://fantasy.premierleague.com/api/entry/{}/event/{}/picks/'
url_wow = str.format(team_id,gameweek)

with urllib.request.urlopen(url_wow) as f5:
    d5 = json.load(f5)

df_wow_performance = pd.json_normalize(d5['picks'])

if gameweek2 != 0:
    url_past_week = str.format(team_id,gameweek2)

    with urllib.request.urlopen(url_past_week) as f:
        d = json.load(f)
    
    df_wow_performance_m1 = pd.json_normalize(d['picks'])
    
# %%

# %%
str = 'https://fantasy.premierleague.com/api/bootstrap-static/'
url_names = str

with urllib.request.urlopen(url_names) as f10:
    d10 = json.load(f10)

df_names = pd.json_normalize(d10['elements'])
df_names_1 = df_names[['id','web_name','element_type','form','value_form','selected_by_percent']]

df_wow_performance.rename(columns={'element':'id'}, inplace=True)

df_wow_performance_with_names = pd.merge(df_wow_performance,df_names_1,how='inner',on='id')

map_pos_dict = {1:'GK',2:'DEF',3:'MID',4:"FWD"}

df_wow_performance_with_names['element_type']=df_wow_performance_with_names['element_type'].map(map_pos_dict)

player_pts = []

str = 'https://fantasy.premierleague.com/api/element-summary/{}/'
for i in list(df_wow_performance_with_names['id']):
    url = str.format(i)

    with urllib.request.urlopen(url) as f1:
        d1=json.load(f1)
    
    df = pd.json_normalize(d1['history'])
    df1 = df[['element','round','total_points','value']]
    df1['season_avg'] = df1['total_points'].mean()
    # df1['avg_pts_3'] = df1['total_points'].tail(3).mean()
    # df1['avg_pts_5'] = df1['total_points'].tail(5).mean()
    df1['rolling_average_2'] = df1['total_points'].rolling(window=2,min_periods=1).mean()
    df1['rolling_average_3'] = df1['total_points'].rolling(window=3,min_periods=1).mean()
    df1['rolling_average_5'] = df1['total_points'].rolling(window=5,min_periods=1).mean()
    df1['total_season_pts'] = df1['total_points'].sum()
    df1.rename(columns={'element':'id'},inplace=True)
    # df1['id'] = i

    player_pts.append(df1)


df_player_pts = pd.concat(player_pts, ignore_index=True)
#%%

df_player_pts_current_gw = df_player_pts[df_player_pts['round'] == gameweek]

df_wow_performance_with_pts = pd.merge(df_wow_performance_with_names, df_player_pts_current_gw,how='inner',on='id')
df_wow_performance_with_pts.rename(columns={'web_name':'player_name','value':'price'}, inplace=True)
df_past_2_games_avg=df_wow_performance_with_pts['rolling_average_2'].mean()

#%%

#Transfer history
df_to_map = df_names_1[['id','web_name']]
df_to_map_prices = df_names[['id','now_cost']]
df_to_map_points = df_names[['id','event_points']]
names_dict = dict(df_to_map.values)
prices_dict = dict(df_to_map_prices.values)
points_dict = dict(df_to_map_points.values)
transfers = list(set(list(df_player_pts_current_gw['id']))-set(list(df_wow_performance_m1['element'])))
transfers_out = list(set(list(df_wow_performance_m1['element']))-set(list(df_player_pts_current_gw['id'])))
df_transfers = pd.DataFrame(transfers,columns=['Players Transferred In'])
df_transfers['Price'] = df_transfers['Players Transferred In']
df_transfers['Points'] = df_transfers['Players Transferred In']
df_transfers_out = pd.DataFrame(transfers_out,columns=['Players Transferred Out'])
df_transfers_out['Price'] = df_transfers_out['Players Transferred Out']
df_transfers_out['Points'] = df_transfers_out['Players Transferred Out']
df_transfers['Players Transferred In'] = df_transfers['Players Transferred In'].map(names_dict)
df_transfers['Price'] = df_transfers['Price'].map(prices_dict)
df_transfers['Points'] = df_transfers['Points'].map(points_dict)
df_transfers_out['Players Transferred Out/No Fixtures'] = df_transfers_out['Players Transferred Out'].map(names_dict)
df_transfers_out['Price'] = df_transfers_out['Price'].map(prices_dict)
df_transfers_out['Points'] = df_transfers_out['Points'].map(points_dict)
net_points = sum(df_transfers['Points']) - sum(df_transfers_out['Points'])
gameweek_points = sum(df_wow_performance_with_pts['total_points'].head(11))

#%%
#Calculate Benchmarks for value_form
#Plot graph for value_form
#Use df_names
df_benchmarks_base = df_names
df_benchmarks_base[['value_form','form','points_per_game']] = df_benchmarks_base[['value_form','form','points_per_game']].apply(pd.to_numeric)
df_benchmarks_clean = df_benchmarks_base.fillna(0)
df_benchmarks_clean.rename(columns={'web_name':'Name'}, inplace=True)
#4.5 mil to 5.5 mil
df_tier_1 = df_benchmarks_clean[df_benchmarks_clean['now_cost'].between(45,55, inclusive=True)]
df_tier_1_value = df_tier_1[['points_per_game']].sort_values(by=['points_per_game'],ascending=[False]).head(50).mean()
df_tier_1_table = df_tier_1[['Name','now_cost','form','points_per_game','value_form','total_points']].sort_values(by=['points_per_game'],ascending=[False]).head(50)
df_tier_1_table['tier'] = '4.5 to 5.5'
#5.5 to 6.5 mil
df_tier_2 = df_benchmarks_clean[df_benchmarks_clean['now_cost'].between(55,65, inclusive=True)]
df_tier_2_value = df_tier_2[['points_per_game']].sort_values(by=['points_per_game'],ascending=[False]).head(50).mean()
df_tier_2_table = df_tier_2[['Name','now_cost','form','points_per_game','value_form','total_points']].sort_values(by=['points_per_game'],ascending=[False]).head(50)
df_tier_2_table['tier'] = '5.5 to 6.5'
#6.5 to 8mil
df_tier_3 = df_benchmarks_clean[df_benchmarks_clean['now_cost'].between(65,80, inclusive=True)]
df_tier_3_value = df_tier_3[['points_per_game']].sort_values(by=['points_per_game'],ascending=[False]).head(50).mean()
df_tier_3_table = df_tier_3[['Name','now_cost','form','points_per_game','value_form','total_points']].sort_values(by=['points_per_game'],ascending=[False]).head(50)
df_tier_3_table['tier'] = '6.5 to 8.0'
#8 to 9 mil 
df_tier_4 = df_benchmarks_clean[df_benchmarks_clean['now_cost'].between(80,90, inclusive=True)]
df_tier_4_value = df_tier_4[['points_per_game']].sort_values(by=['points_per_game'],ascending=[False]).head(50).mean()
df_tier_4_table = df_tier_4[['Name','now_cost','form','points_per_game','value_form','total_points']].sort_values(by=['points_per_game'],ascending=[False]).head(50)
df_tier_4_table['tier'] = '8.0 to 9.0'
#9 to 12 mil 
df_tier_5 = df_benchmarks_clean[df_benchmarks_clean['now_cost'].between(90,120, inclusive=True)]
df_tier_5_value = df_tier_5[['points_per_game']].sort_values(by=['points_per_game'],ascending=[False]).head(50).mean()
df_tier_5_table = df_tier_5[['Name','now_cost','form','points_per_game','value_form','total_points']].sort_values(by=['points_per_game'],ascending=[False]).head(50)
df_tier_5_table['tier'] = '9.0 to 12.0'
#12 mil and above
df_tier_6 = df_benchmarks_clean[df_benchmarks_clean['now_cost'] >= 120]
df_tier_6_value = df_tier_6[['points_per_game']].sort_values(by=['points_per_game'],ascending=[False]).head(50).mean()
df_tier_6_table = df_tier_6[['Name','now_cost','form','points_per_game','value_form','total_points']].sort_values(by=['points_per_game'],ascending=[False]).head(50)
df_tier_6_table['tier'] = '>12.0'

#All tables together
tier_all = [df_tier_1_table,df_tier_2_table,df_tier_3_table,df_tier_4_table,df_tier_5_table,df_tier_6_table]
df_tier_all_table = pd.concat(tier_all)


#%%
#Get xPts Table
df_expected_points = pd.read_csv('https://raw.githubusercontent.com/theFPLkiwi/webpage/main/data/Projected_FPL_2223.csv')
df_expected_points_1 = df_expected_points[['Name','Pos','Price','Team']]
start = list(df_expected_points.columns).index('xPts if play')
start_1 = start+1
end = start_1 +5
df_expected_points_2 = df_expected_points[list(df_expected_points.columns)[start_1:end]]
df_expected_points_clean = pd.concat([df_expected_points_1,df_expected_points_2], axis=1)
#%%
# df_expected_points_clean = df_expected_points[['Name','Pos','Price','Team','8.1',
#  '9.1',
#  '10.1',
#  '11.1',
#  '12.1',
#  '13.1',
#  '14.1',
#  '15.1',
#  '16.1',
#  '17.1']]
df_expected_points_clean['Average Expected Points/Game (Over 5 Games)']=df_expected_points_clean[list(df_expected_points_clean.columns)[4:9]].mean(axis=1)
df_expected_points_clean['Expected Points/Game Per Mil'] = df_expected_points_clean['Average Expected Points/Game (Over 5 Games)']/df_expected_points_clean['Price']
#%%
#4.5 mil to 5.5 mil Future
df_tier_1_future_table = df_expected_points_clean[df_expected_points_clean['Price'].between(4.5,5.5,inclusive=True)].head(50)
df_tier_1_future_table['tier'] = '4.5 to 5.5'

#5.5 mil to 6.5 mil Future
df_tier_2_future_table = df_expected_points_clean[df_expected_points_clean['Price'].between(5.6,6.5,inclusive=True)].head(50)
df_tier_2_future_table['tier'] = '5.5 to 6.5'

#6.5 mil to 8.0 mil Future
df_tier_3_future_table = df_expected_points_clean[df_expected_points_clean['Price'].between(6.6,8.0,inclusive=True)].head(50)
df_tier_3_future_table['tier'] = '6.5 to 8.0'

#8.0 mil to 9.0 mil Future
df_tier_4_future_table = df_expected_points_clean[df_expected_points_clean['Price'].between(8.1,9.0,inclusive=True)].head(50)
df_tier_4_future_table['tier'] = '8.0 to 9.0'

#9.0 mil to 12.0 mil Future
df_tier_5_future_table = df_expected_points_clean[df_expected_points_clean['Price'].between(9.1,12.0,inclusive=True)].head(50)
df_tier_5_future_table['tier'] = '9.0 to 12.0'

#>12 Future
df_tier_6_future_table = df_expected_points_clean[df_expected_points_clean['Price']> 12.0].head(50)
df_tier_6_future_table['tier'] = '>12.0'


#Merge All Tables With Average Upcoming Points
tier_future_all = [df_tier_1_future_table,df_tier_2_future_table,df_tier_3_future_table,df_tier_4_future_table,df_tier_5_future_table,df_tier_6_future_table]
df_tier_all_future_table = pd.concat(tier_future_all)

#Suggested Transfers
df_suggested_transfers = df_wow_performance_with_pts[df_wow_performance_with_pts['rolling_average_2'] < 3][['player_name','price','season_avg','rolling_average_2','rolling_average_5','total_season_pts']].sort_values(by=['rolling_average_2'],ascending=True).head(5)
df_suggested_transfers_in = df_tier_all_future_table.sort_values(by=['Average Expected Points/Game (Over 5 Games)'],ascending=False)

#%%
#Understat Data
url_understat = 'https://understat.com/league/EPL/2022'

res = requests.get(url_understat)
soup = BeautifulSoup(res.content,'lxml')
scripts = soup.find_all('script')
#get exact table
strings = scripts[3].string
#strip to only JSON data
ind_start = strings.index("('")+2
ind_end = strings.index("')")
#strip unnecessary stuff
json_data = strings[ind_start:ind_end]
json_data = json_data.encode('utf8').decode('unicode_escape')
#convert string to json
data = json.loads(json_data)
df_understat = pd.json_normalize(data)
#cleaning data
df_understat[['id','games', 'time', 'goals', 'xG', 'assists', 'xA',
   'shots', 'key_passes', 'yellow_cards', 'red_cards','npg', 'npxG', 'xGChain', 'xGBuildup']]=df_understat[['id','games', 'time', 'goals', 'xG', 'assists', 'xA',
   'shots', 'key_passes', 'yellow_cards', 'red_cards','npg', 'npxG', 'xGChain', 'xGBuildup']].apply(pd.to_numeric)
df_understat[['player_name','position','team_title']] = df_understat[['player_name','position','team_title']].convert_dtypes()
df_understat['90s'] = df_understat['time']/90
df_understat_clean = df_understat[df_understat['90s'] > 5]
df_understat_clean['xG/90'] = df_understat_clean['xG']/df_understat_clean['90s']
df_understat_clean['Shots/90'] = df_understat_clean['shots']/df_understat_clean['90s']
df_understat_clean['xA/90'] = df_understat_clean['xA']/df_understat_clean['90s']
df_understat_clean['Key Passes/90'] = df_understat_clean['key_passes']/df_understat_clean['90s']
df_understat_clean['xG+xA per 90'] = df_understat_clean['xG/90'] + df_understat_clean['xA/90']
df_understat_table = df_understat_clean[['player_name'] + list(df_understat_clean.columns)[18:24]]
df_understat_table = df_understat_table.fillna(0)
# df_understat_table.sort_values(by=['xG+xA per 90'],ascending=[False],inplace=True)
#%%

col1,col2,col3 = st.columns(3)

def highlight_cols(s):
    color = 'green'
    return 'background-color: %s' % color

with col1:
    st.table(df_transfers)
with col2:
    st.table(df_transfers_out[['Players Transferred Out/No Fixtures','Price','Points']])
st.text(f'Net Points From Transfers: {net_points}')
st.text(f'Average Points/Game Past 2 Games: {round(df_past_2_games_avg,2)}')
st.text(f'Total Gameweek Points: {gameweek_points}')
st.text('Note: value_form and selected_by_percent reflect latest gameweek values, not retrospective values')
with st.expander('⬇️ Column Definitions'):
    st.text('value_form = Player price divided by recent form. This measures efficiency of spend.')
    st.text('rolling _average = The average points over the past 2, 3, or 5 games from the selected gameweek.')
st.table(df_wow_performance_with_pts.drop(columns=['id','form','position','multiplier','is_vice_captain']).style.background_gradient(cmap='RdYlGn', subset=pd.IndexSlice[:,['value_form','total_points','rolling_average_2']]).set_precision(2))
st.subheader('Underperforming Players To Consider Transferring Out')
st.table(df_suggested_transfers)
st.subheader('Top 10 Suggested Transfers In')
st.text('For Latest GW Only, Data Is Not Retrospective')
col1,col2,col3 = st.columns(3)

with col1:
    tier = st.multiselect(
        'Select Price Tier',
        options=df_suggested_transfers_in['tier'].unique(),
        default= '6.5 to 8.0'
    )
    all_tiers = st.checkbox('Select All Tiers (Uncheck when filtering)',value =False)

    if all_tiers:
        tier = list(df_suggested_transfers_in['tier'].unique())

with col2:
    team = st.multiselect(
        'Select Team',
        options=df_suggested_transfers_in['Team'].unique(),
    )
    all_teams = st.checkbox('Select All Teams (Uncheck when filtering)',value =True)

    if all_teams:
        team = list(df_suggested_transfers_in['Team'].unique())

with col3:
    pos = st.multiselect(
        'Select Position',
        options=df_suggested_transfers_in['Pos'].unique(),
    )

    all_pos = st.checkbox('Select All Pos (Uncheck when filtering)',value =True)

    if all_pos:
        pos = list(df_suggested_transfers_in['Pos'].unique())

df_suggested_transfers_in_query = df_suggested_transfers_in.query(
    "tier == @tier and Pos == @pos and Team == @team"
)
df_suggested_transfers_in_query.columns = df_suggested_transfers_in_query.columns.str.replace('.1',' ')
st.table(df_suggested_transfers_in_query.head(10).style.background_gradient(cmap='RdYlGn').set_precision(2))

st.subheader('OPTA Stats')
st.text('Validate Your Choices With Underlying Stats!')

player1= st.multiselect(
    'Select Players',
    options=df_understat_table['player_name'].unique(),
    default=list(df_understat_table.head(10)['player_name'].unique())  
)

df_opta_table = df_understat_table.query(
    "player_name == @player1"
)
st.table(df_opta_table.sort_values(by=['xG+xA per 90'],ascending=[False]).style.background_gradient(cmap='RdYlGn').set_precision(2))

st.subheader('Misc Stats & Trends')
st.text('Points Per Game by Price Tiers - Which Tier Has Been Driving Value?')
fig = px.violin(df_tier_all_table, y='points_per_game',hover_data=['Name'],points='all',box=True,color='tier')
fig.update_layout(legend=dict(orientation='h'), margin=dict(l=10, r=10, t=10, b=10),modebar_remove=['zoom', 'select'],dragmode = False)
st.plotly_chart(fig, use_container_width=True)

col1,col2,col3 = st.columns(3)
with col1:
    st.text(f'4.5m-5.5m Top 50 PPG Benchmark: {round(df_tier_1_value[0],2)}')
    st.dataframe(df_tier_1_table)

with col2:
    st.text(f'5.5m-6.5m Top 50 PPG Benchmark: {round(df_tier_2_value[0],2)}')
    st.dataframe(df_tier_2_table)

with col3:
    st.text(f'6.5m-8.0m Top 50 PPG Benchmark: {round(df_tier_3_value[0],2)}')
    st.dataframe(df_tier_3_table)

col1,col2,col3 = st.columns(3)
with col1:
    st.text(f'8.0m-9.0m Top 50 PPG Benchmark: {round(df_tier_4_value[0],2)}')
    st.dataframe(df_tier_4_table)

with col2:
    st.text(f'9.0m-11.0m Top 50 PPG Benchmark: {round(df_tier_5_value[0],2)}')
    st.dataframe(df_tier_5_table)

with col3:
    st.text(f'>12.0m Top 50 PPG Benchmark: {round(df_tier_6_value[0],2)}')
    st.dataframe(df_tier_6_table)

st.subheader('Expected Points')
fig2 = px.violin(df_tier_all_future_table, y='Average Expected Points/Game (Over 5 Games)',hover_data=['Name'],points='all',box=True,color='tier')
fig2.update_layout(legend=dict(orientation='h'), margin=dict(l=10, r=10, t=10, b=10),modebar_remove=['zoom', 'select'],dragmode = False)
st.plotly_chart(fig2, use_container_width=True)
future_player = st.multiselect(
    'Select Players To Compare',
    options=df_expected_points_clean['Name'].unique()
)

all_fut_player = st.checkbox('Select All Players (Uncheck when filtering)',value =True)

if all_fut_player:
    future_player = list(df_expected_points_clean['Name'].unique())

df_expected_points_clean_query = df_expected_points_clean.query(
    "Name == @future_player"
)
st.dataframe(df_expected_points_clean_query.sort_values(by=['Average Expected Points/Game (Over 5 Games)'],ascending=[False]).style.background_gradient(cmap='RdYlGn', subset=pd.IndexSlice[:,['Average Expected Points/Game (Over 5 Games)','Expected Points/Game Per Mil']]).set_precision(2))

#We need to get element names -> Dictionary of names
   #We can this from bootstrap-static -> elements -> get both id and web_name
   #Merge with df_wow_performance table
   #DONE
#We need to get element types 
    #We can this from bootstrap-static -> element_types
    #DONE
#We need to get element points, element past 5 games
    #We can get this from element-summary -> history
    #Get last 5 of dataframe -> get sum of total_points -> sum of total points / len(last 5 of dataframe) -> merge with df_wow_performance table
    #DONE
#We need to find a way to predict macro price increases -> study this
#We need a benchmark calculator, and to sort PPG by cost
    #Calculate average points of 4.5 mil players, those that played more than 45min-> Points-average points/Price-4.5mil
    #Calculate benchmark based on price
#We also need a way to calculate expected gain in differential points
    #
#Have a view of 5 week rolling mean of competitor players as well as a w-o-w view, compare vs other players
#Add transfer history by week
#Add xG, xA, Big Chances, Clean Sheets, Attempted Assists
#Add Rank Graph
#Add Benchmark Graphs
#Replace suggested transfers with expected point table, with ability to filter. Next to it, add xG Table with ability to filter player names
# %%
