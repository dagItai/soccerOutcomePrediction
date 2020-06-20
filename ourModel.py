import xmltodict
import json
import sqlite3
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

#  ========== Connection ==========
path = "C:\Users\itai.dagan\Documents\PredictionModel"  # Insert path here
database = path + '\database.sqlite'
conn = sqlite3.connect(database)

# ========== Extract Relevant Data From DB ==========
# Data
data = pd.read_sql("""SELECT Match.id, 
                                        League.name AS league_name, 
                                        season, 
                                        stage, 
                                        date,
                                        shoton,
                                        shotoff,
                                        goal,
                                        corner,
                                        foulcommit,
                                        card,
                                        HT.team_long_name AS  home_team,
                                        AT.team_long_name AS away_team,
                                        HT.team_api_id AS home_team_api_id,
                                        AT.team_api_id AS away_team_api_id,
                                        home_team_goal, 
                                        away_team_goal,
                                        home_player_1, 
                                        home_player_2,
                                        home_player_3, 
                                        home_player_4, 
                                        home_player_5, 
                                        home_player_6, 
                                        home_player_7, 
                                        home_player_8, 
                                        home_player_9, 
                                        home_player_10, 
                                        home_player_11, 
                                        away_player_1, 
                                        away_player_2, 
                                        away_player_3, 
                                        away_player_4, 
                                        away_player_5, 
                                        away_player_6, 
                                        away_player_7, 
                                        away_player_8, 
                                        away_player_9, 
                                        away_player_10, 
                                        away_player_11
                                FROM Match
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE season not like '2015/2016' and goal is not null
                                ORDER by date
                                LIMIT 100000;""", conn)

player_stats_data = pd.read_sql("SELECT player_api_id,overall_rating,strength FROM Player_Attributes;", conn)
player_stats_data = player_stats_data.groupby(['player_api_id']).mean()
data_test = pd.read_sql("""SELECT Match.id, 
                                        League.name AS league_name, 
                                        season, 
                                        stage, 
                                        date,
                                        shoton,
                                        shotoff,
                                        goal,
                                        corner,
                                        foulcommit,
                                        card,
                                        HT.team_long_name AS  home_team,
                                        AT.team_long_name AS away_team,
                                        HT.team_api_id AS home_team_api_id,
                                        AT.team_api_id AS away_team_api_id,
                                        home_team_goal, 
                                        away_team_goal,
                                        home_player_1, 
                                        home_player_2,
                                        home_player_3, 
                                        home_player_4, 
                                        home_player_5, 
                                        home_player_6, 
                                        home_player_7, 
                                        home_player_8, 
                                        home_player_9, 
                                        home_player_10, 
                                        home_player_11, 
                                        away_player_1, 
                                        away_player_2, 
                                        away_player_3, 
                                        away_player_4, 
                                        away_player_5, 
                                        away_player_6, 
                                        away_player_7, 
                                        away_player_8, 
                                        away_player_9, 
                                        away_player_10, 
                                        away_player_11                                      
                                FROM Match
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE season like '2015/2016' 
                                ORDER by date
                                LIMIT 100000;""", conn)


# ========== Help Functions ==========
def convert_xml_to_json(xml):
    if xml == None:
        return json.dumps({})
    my_dict = xmltodict.parse(xml)
    json_data = json.dumps(my_dict)
    return json_data

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# ========== Data Preparation ==========
# Processing team shots
def team_shots(l, data_local, home_team_shots_local, away_team_shots_local):
    home_team_shot = 0
    away_team_shot = 0
    hasVal = False
    try:
        jsonObj = json.loads(convert_xml_to_json(data_local.shotoff[l]))['shotoff']
        if type(jsonObj['value']) == list:
            obj = jsonObj['value']
            for i in obj:
                if 'team' in i.keys():
                    hasVal = True
                    team = int(i['team'])
                    if team == data_local.away_team_api_id[l]:
                        away_team_shot += 1
                    else:
                        home_team_shot += 1
        else:
            obj = jsonObj['value']
            if 'team' in obj.keys():
                hasVal = True
                team = int(obj['team'])
                if team == data_local.away_team_api_id[l]:
                    away_team_shot += 1
                else:
                    home_team_shot += 1
    except:
        home_team_shots_local.append(np.NaN)
        away_team_shots_local.append(np.NaN)

    if hasVal:
        home_team_shots_local.append(home_team_shot)
        away_team_shots_local.append(away_team_shot)


# Processing team shots to the target
def team_shots_target(l, data_local, home_team_shots_target_local, away_team_shots_target_local):
    home_team_shot_target = 0
    away_team_shot_target = 0
    hasVal = False
    try:
        jsonObj = json.loads(convert_xml_to_json(data_local.shoton[l]))['shoton']
        if type(jsonObj['value']) == list:
            obj = jsonObj['value']
            for i in obj:
                if 'team' in i.keys():
                    hasVal = True
                    team = int(i['team'])
                    if team == data_local.away_team_api_id[l]:
                        away_team_shot_target += 1
                    else:
                        home_team_shot_target += 1
        else:
            obj = jsonObj['value']
            if 'team' in obj.keys():
                hasVal = True
                team = int(obj['team'])
                if team == data_local.away_team_api_id[l]:
                    away_team_shot_target += 1
                else:
                    home_team_shot_target += 1
    except:
        home_team_shots_target_local.append(np.NaN)
        away_team_shots_target_local.append(np.NaN)

    if hasVal:
        home_team_shots_target_local.append(home_team_shot_target)
        away_team_shots_target_local.append(away_team_shot_target)


# Processing team half time goal - all the goals before the half time
def team_half_time_goals(l, data_local, home_team_half_time_goals_local, away_team_half_time_goals_local):
    half_time_home_team_goal = 0
    half_time_away_team_goal = 0
    hasVal = False
    try:
        jsonObj = json.loads(convert_xml_to_json(data_local.goal[l]))['goal']
        if type(jsonObj['value']) == list:
            obj = jsonObj['value']
            for i in obj:
                if 'team' in i.keys() and 'elapsed' in i.keys():
                    if int(i['elapsed']) <= 45:
                        hasVal = True
                        team = int(i['team'])
                        if team == data_local.away_team_api_id[l]:
                            half_time_away_team_goal += 1
                        else:
                            half_time_home_team_goal += 1
            if not hasVal:
                home_team_half_time_goals_local.append(np.NaN)
                away_team_half_time_goals_local.append(np.NaN)
        else:
            obj = jsonObj['value']
            if 'team' in obj.keys() and 'elapsed' in obj.keys():
                if int(obj['elapsed']) <= 45:
                    hasVal = True
                    team = int(obj['team'])
                    if team == data_local.away_team_api_id[l]:
                        half_time_away_team_goal += 1
                    else:
                        half_time_home_team_goal += 1
            if not hasVal:
                home_team_half_time_goals_local.append(np.NaN)
                away_team_half_time_goals_local.append(np.NaN)
    except:
        home_team_half_time_goals_local.append(np.NaN)
        away_team_half_time_goals_local.append(np.NaN)

    if hasVal:
        home_team_half_time_goals_local.append(half_time_home_team_goal)
        away_team_half_time_goals_local.append(half_time_away_team_goal)


# Processing all the team cards during the game yellow and red separably
def team_cards(l, data_local, away_team_yellows_local, home_team_yellows_local, away_team_reds_local,
               home_team_reds_local):
    away_team_yellow = 0
    home_team_yellow = 0
    away_team_red = 0
    home_team_red = 0
    jsonObj = json.loads(convert_xml_to_json(data_local.card[l]))
    hasVal = False
    if jsonObj is not None:
        if 'card' in jsonObj.keys():
            jsonObj = jsonObj['card']
            if jsonObj is not None:
                if type(jsonObj['value']) == list:
                    for i in jsonObj['value']:
                        if 'card_type' in i.keys():
                            card_type = i['card_type']
                        else:
                            card_type = i['comment']
                        if 'team' in i.keys():
                            hasVal = True
                            team = int(i['team'])
                            if team == data_local.away_team_api_id[l]:
                                if card_type == 'y' or card_type == 'y2':
                                    away_team_yellow += 1
                                else:
                                    away_team_red += 1
                            else:
                                if card_type == 'y' or card_type == 'y2':
                                    home_team_yellow += 1
                                else:
                                    home_team_red += 1
                else:
                    if 'card_type' in jsonObj['value'].keys():
                        card_type = jsonObj['value']['card_type']
                    else:
                        card_type = jsonObj['value']['comment']
                    if 'team' in jsonObj['value'].keys():
                        hasVal = True
                        team = int(jsonObj['value']['team'])
                        if team == data_local.away_team_api_id[l]:
                            if card_type == 'y' or card_type == 'y2':
                                away_team_yellow += 1
                            else:
                                away_team_red += 1
                        else:
                            if card_type == 'y' or card_type == 'y2':
                                home_team_yellow += 1
                            else:
                                home_team_red += 1
    if hasVal:
        away_team_yellows_local.append(away_team_yellow)
        home_team_yellows_local.append(home_team_yellow)
        away_team_reds_local.append(away_team_red)
        home_team_reds_local.append(home_team_red)
    else:
        away_team_yellows_local.append(np.NaN)
        home_team_yellows_local.append(np.NaN)
        away_team_reds_local.append(np.NaN)
        home_team_reds_local.append(np.NaN)


# Processing how many fouls each team has made
def team_fouls_committed(l, data_local, home_f_local, away_f_local):
    if data_local.foulcommit[l] is not None:
        countf_home = 0
        countf_away = 0
        jsonObjf = json.loads(convert_xml_to_json(data_local.foulcommit[l]))
        if jsonObjf is not None:
            if 'foulcommit' in jsonObjf:
                if jsonObjf['foulcommit'] is not None:
                    if type(jsonObjf['foulcommit']['value']) == list:
                        for i in jsonObjf['foulcommit']['value']:
                            if 'team' in i.keys():
                                team2 = i['team']
                                home2 = data_local.home_team_api_id[l]
                                if int(team2) == home2:
                                    countf_home += 1
                                else:
                                    countf_away += 1
            else:
                countf_home = np.NaN
                countf_away = np.NaN
        home_f_local.append(countf_home)
        away_f_local.append(countf_away)
    else:
        home_f_local.append(np.NaN)
        away_f_local.append(np.NaN)


# Processing how many corners each team made
def team_corners(l, data_local, home_corner_local, away_corner_local):
    if data_local.corner[l] is not None:
        count_home = 0
        count_away = 0
        jsonObj = json.loads(convert_xml_to_json(data_local.corner[l]))
        if jsonObj is not None:
            if 'corner' in jsonObj:
                if jsonObj['corner'] is not None:
                    if type(jsonObj['corner']['value']) == list:
                        for i in jsonObj['corner']['value']:
                            if 'team' in i.keys():
                                team = i['team']
                                home = data_local.home_team_api_id[l]
                                if int(team) == home:
                                    count_home += 1
                                else:
                                    count_away += 1
            else:
                count_home = np.NaN
                count_away = np.NaN
        home_corner_local.append(count_home)
        away_corner_local.append(count_away)
    else:
        home_corner_local.append(np.NaN)
        away_corner_local.append(np.NaN)


# Fill all the columns with NA fields with the mean of the column
def fillNA(data_local):
    data_local['AY'].fillna((data_local['AY'].mean()), inplace=True)
    data_local['HY'].fillna((data_local['HY'].mean()), inplace=True)
    data_local['AR'].fillna((data_local['AR'].mean()), inplace=True)
    data_local['HR'].fillna((data_local['HR'].mean()), inplace=True)

    data_local['HS'].fillna((data_local['HS'].mean()), inplace=True)
    data_local['AS'].fillna((data_local['AS'].mean()), inplace=True)
    data_local['HST'].fillna((data_local['HST'].mean()), inplace=True)
    data_local['AST'].fillna((data_local['AST'].mean()), inplace=True)
    data_local['HTHG'].fillna((data_local['HTHG'].mean()), inplace=True)
    data_local['HTAG'].fillna((data_local['HTAG'].mean()), inplace=True)

    data_local['HC'].fillna((data_local['HC'].mean()), inplace=True)
    data_local['AC'].fillna((data_local['AC'].mean()), inplace=True)
    data_local['HF'].fillna((data_local['HF'].mean()), inplace=True)
    data_local['AF'].fillna((data_local['AF'].mean()), inplace=True)

    data_local['HPS'].fillna((data_local['HPS'].mean()), inplace=True)
    data_local['APS'].fillna((data_local['APS'].mean()), inplace=True)
    data_local['HPR'].fillna((data_local['HPR'].mean()), inplace=True)
    data_local['APR'].fillna((data_local['APR'].mean()), inplace=True)


# Calculate the away and home team average overall_rating and their strength
def getTeamPlayersStats(l, data_local, home_players_strength, away_players_strength, home_players_overall,
                        away_players_overall):
    home_players_local = []
    away_players_local = []
    data_row = data_local.loc[[l]]
    home_players_local.append(data_row.home_player_1)
    home_players_local.append(data_row.home_player_2)
    home_players_local.append(data_row.home_player_3)
    home_players_local.append(data_row.home_player_4)
    home_players_local.append(data_row.home_player_5)
    home_players_local.append(data_row.home_player_6)
    home_players_local.append(data_row.home_player_7)
    home_players_local.append(data_row.home_player_8)
    home_players_local.append(data_row.home_player_9)
    home_players_local.append(data_row.home_player_10)
    home_players_local.append(data_row.home_player_11)
    away_players_local.append(data_row.away_player_1)
    away_players_local.append(data_row.away_player_2)
    away_players_local.append(data_row.away_player_3)
    away_players_local.append(data_row.away_player_4)
    away_players_local.append(data_row.away_player_5)
    away_players_local.append(data_row.away_player_6)
    away_players_local.append(data_row.away_player_7)
    away_players_local.append(data_row.away_player_8)
    away_players_local.append(data_row.away_player_9)
    away_players_local.append(data_row.away_player_10)
    away_players_local.append(data_row.away_player_11)
    overall_rating = 0
    overall_rating_counter = 0
    strength = 0
    strength_counter = 0
    for player in home_players_local:
        if not math.isnan(player.real[0]):
            row = player_stats_data.loc[[int(player.real[0])]]
            if 'overall_rating' in row.keys():
                overall_rating += row['overall_rating'].real[0]
                overall_rating_counter += 1
            if 'strength' in row.keys():
                strength += row['strength'].real[0]
                strength_counter += 1
    if strength_counter > 0:
        strength = strength / strength_counter
    else:
        strength = np.NaN
    if overall_rating_counter > 0:
        overall_rating = overall_rating / overall_rating_counter
    else:
        overall_rating = np.NaN
    home_players_strength.append(strength)
    home_players_overall.append(overall_rating)
    overall_rating = 0
    overall_rating_counter = 0
    strength = 0
    strength_counter = 0
    for player in away_players_local:
        if not math.isnan(player.real[0]):
            row = player_stats_data.loc[[int(player.real[0])]]
            if 'overall_rating' in row.keys():
                overall_rating += row['overall_rating'].real[0]
                overall_rating_counter += 1
            if 'strength' in row.keys():
                strength += row['strength'].real[0]
                strength_counter += 1
    if strength_counter > 0:
        strength = strength / strength_counter
    else:
        strength = np.NaN
    if overall_rating_counter > 0:
        overall_rating = overall_rating / overall_rating_counter
    else:
        overall_rating = np.NaN
    away_players_strength.append(strength)
    away_players_overall.append(overall_rating)


# Defining all the vectors that has to be field
away_team_yellows = []
home_team_yellows = []
away_team_reds = []
home_team_reds = []
# ===================
home_corner = []
away_corner = []
home_f = []
away_f = []
# ===================
home_team_shots_target = []
away_team_shots_target = []
home_team_half_time_goals = []
away_team_half_time_goals = []
home_team_shots = []
away_team_shots = []
# ==================
win = []
# ==================
home_players_strength = []
away_players_strength = []
home_players_overall = []
away_players_overall = []

# Main Loop to iterate over the data row and manipulate the rows
for l in range(0, len(data)):
    team_shots(l, data, home_team_shots, away_team_shots)
    team_shots_target(l, data, home_team_shots_target, away_team_shots_target)
    team_half_time_goals(l, data, home_team_half_time_goals, away_team_half_time_goals)
    team_cards(l, data, away_team_yellows, home_team_yellows, away_team_reds, home_team_reds)
    team_fouls_committed(l, data, home_f, away_f)
    team_corners(l, data, home_corner, away_corner)
    getTeamPlayersStats(l, data, home_players_strength, away_players_strength, home_players_overall,
                        away_players_overall)
    # Define the Winner - If Home Team is the winner, win => 1, otherwise win => 0
    if data.home_team_goal[l] > data.away_team_goal[l]:
        k1 = 1
        win.append(k1)
    else:
        k1 = 0
        win.append(k1)

# Create Data Frame of the training
data = pd.DataFrame({
    "HomeTeam": data.home_team_api_id,
    "AwayTeam": data.away_team_api_id,
    "FTHG": data.home_team_goal,
    "FTAG": data.away_team_goal,
    "AY": away_team_yellows,
    "HY": home_team_yellows,
    "AR": away_team_reds,
    "HR": home_team_reds,
    "HS": home_team_shots,
    "AS": away_team_shots,
    "HST": home_team_shots_target,
    "AST": away_team_shots_target,
    "HTHG": home_team_half_time_goals,
    "HTAG": away_team_half_time_goals,
    "HC": home_corner,
    "AC": away_corner,
    "HF": home_f,
    "AF": away_f,
    "HPR": home_players_overall,
    "APR": away_players_overall,
    "HPS": home_players_strength,
    "APS": away_players_strength,
    "winner": win
})

# Fill NA With Mean Value of the training
fillNA(data)

#  vectors
away_team_yellows = []
home_team_yellows = []
away_team_reds = []
home_team_reds = []
# ===================
home_corner = []
away_corner = []
home_f = []
away_f = []
# ===================
home_team_shots_target = []
away_team_shots_target = []
home_team_half_time_goals = []
away_team_half_time_goals = []
home_team_shots = []
away_team_shots = []
# ==================
win_test = []
# ==================
home_players_strength = []
away_players_strength = []
home_players_overall = []
away_players_overall = []

for l in range(0, len(data_test)):
    team_shots(l, data_test, home_team_shots, away_team_shots)
    team_shots_target(l, data_test, home_team_shots_target, away_team_shots_target)
    team_half_time_goals(l, data_test, home_team_half_time_goals, away_team_half_time_goals)
    team_cards(l, data_test, away_team_yellows, home_team_yellows, away_team_reds, home_team_reds)
    team_fouls_committed(l, data_test, home_f, away_f)
    team_corners(l, data_test, home_corner, away_corner)
    getTeamPlayersStats(l, data_test, home_players_strength, away_players_strength, home_players_overall,
                        away_players_overall)
    # Define the Winner - If Home Team is the winner, win => 1, otherwise win => 0
    if data_test.home_team_goal[l] > data_test.away_team_goal[l]:
        k1 = 1
        win_test.append(k1)
    else:
        k1 = 0
        win_test.append(k1)

# Create Data Frame
data_test = pd.DataFrame({
    "HomeTeam": data_test.home_team_api_id,
    "AwayTeam": data_test.away_team_api_id,
    "FTHG": data_test.home_team_goal,
    "FTAG": data_test.away_team_goal,
    "AY": away_team_yellows,
    "HY": home_team_yellows,
    "AR": away_team_reds,
    "HR": home_team_reds,
    "HS": home_team_shots,
    "AS": away_team_shots,
    "HST": home_team_shots_target,
    "AST": away_team_shots_target,
    "HTHG": home_team_half_time_goals,
    "HTAG": away_team_half_time_goals,
    "HC": home_corner,
    "AC": away_corner,
    "HF": home_f,
    "AF": away_f,
    "HPR": home_players_overall,
    "APR": away_players_overall,
    "HPS": home_players_strength,
    "APS": away_players_strength
})
# Fill NA With Mean Value
fillNA(data_test)

# ============================== Features selection =============================================
X = data.iloc[:, 0:22]  # independent columns
y = data.iloc[:, 22]  # target column i.e price range

array = data.values
X_values = array[:, 0:22]
Y_values = array[:, 22]

model = RandomForestClassifier()
model.fit(X_values, Y_values)
print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(22).plot(kind='barh')
plt.savefig("features.png")

# ============================== Features selection =============================================
# Create DataFrame with the relevant feathers according to the article
df = pd.DataFrame({
    "HS": data.HS,
    "AS": data.AS,
    "HST": data.HST,
    "AST": data.AST,
    "HC": data.HC,
    "HF": data.HF,
    "AF": data.AF,
    "AY": data.AY,
    "HTAG": data.HTAG,
    "HTHG": data.HTHG,
    "HPR": data.HPR,
    "APR": data.APR,
    "HPS": data.HPS,
    "APS": data.APS,
    "winner": win
})

df_test = pd.DataFrame({
    "HS": data_test.HS,
    "AS": data_test.AS,
    "HST": data_test.HST,
    "AST": data_test.AST,
    "HC": data_test.HC,
    "HF": data_test.HF,
    "AF": data_test.AF,
    "AY": data_test.AY,
    "HTAG": data_test.HTAG,
    "HTHG": data_test.HTHG,
    "HPR": data_test.HPR,
    "APR": data_test.APR,
    "HPS": data_test.HPS,
    "APS": data_test.APS,
    "winner": win_test
})

# Compare Algorithms
features_number = 14
# load dataset
array = df.values
X = array[:, 0:features_number]
Y = array[:, features_number]

# prepare configuration for cross validation test harness
seed = 7

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

chosen_model = None
chosen_model_accuracy_score = 0
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
fig.savefig("algorithm.png")

# ========== Hyper-parameter Tuning Using Grid Search & Train the Logistic model with the train set==========
logistic = LogisticRegression()
# Create regularization penalty space
penalty = ['l1', 'l2']
# Create regularization solver space
solver = ['liblinear', 'saga']
# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty, solver=solver)

# Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# Fit grid search
best_model = clf.fit(X, Y)

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best Solver:', best_model.best_estimator_.get_params()['solver'])

# ========== Prepare the test for the model to predict ==========
array = df_test.values
X_test = array[:, 0:features_number]
Y_test = array[:, features_number]

res_pred = best_model.predict(X_test)
score = accuracy_score(Y_test, res_pred)
msg = "%s: %f" % ("LR", score)
# print the classification report for the prediction
print(msg)
print classification_report(Y_test, res_pred)

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, res_pred)
np.set_printoptions(precision=2)

class_names = ['win', 'not win']

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.savefig('conf')  # doctest: +SKIP

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('conf_normalized')  # doctest: +SKIP
