# loadimport os
import json
import requests
from datetime import datetime, timedelta
import numpy as np
import os
import pandas as pd


# Use absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIXTURES_DIR = os.path.join(DATA_DIR, "fixtures")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
TEAM_STATS_DIR = os.path.join(DATA_DIR, "team_stats")
ODDS_DIR = os.path.join(DATA_DIR, "odds")

def fetch_json(url, headers):
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("response", [])
    else:
        print(f"❌ Request failed: {url} ({r.status_code})")
        return []

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_fixtures_stats_target(json_file_path: str, 
                              fixture_id,
                              num_fixtures: int = 10) -> dict:
    """
    Extrae los N fixtures más recientes ordenados por fecha que ocurrieron antes del fixture_id dado
    
    Args:
        json_file_path: Ruta al archivo JSON
        fixture_id: ID del fixture de referencia
        num_fixtures: Número de fixtures a extraer (default 10)
    
    Returns:
        Diccionario con los fixtures más recientes anteriores al fixture_id dado
    """
    
    # Cargar JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Convertir a lista y ordenar por fecha
    fixtures_list = []
    fixtures_id_list = []
    target_fixture_date = None

    # print(data)
    
    for fid, fixture_data in data.items():
        fixtures_id_list.append(fid)
        fixture_data['fixture_id'] = fid
        fixtures_list.append(fixture_data)
        
        # Guardar la fecha del fixture_id de referencia
        if fid == fixture_id:
            target_fixture_date = datetime.strptime(fixture_data['date'], '%Y-%m-%d')
    
    if fixture_id not in fixtures_id_list:
        return None, None
    
    # Filtrar solo fixtures que ocurrieron antes del fixture_id
    x = [
        f for f in fixtures_list 
        if datetime.strptime(f['date'], '%Y-%m-%d') < target_fixture_date
    ]


    
    # Ordenar por fecha descendente (más recientes primero)
    x.sort(
        key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'),
        reverse=True
    )

    x = x[:num_fixtures]

    y = [
        f for f in fixtures_list 
        if datetime.strptime(f['date'], '%Y-%m-%d') == target_fixture_date
    ]
    
    # Tomar los N más recientes
    

     
    
    # Convertir de vuelta a diccionario
    # result = {}
    # for fixture in recent_fixtures:
    #     fid = fixture.pop('fixture_id')
    #     result[fid] = fixture
    
    return x, y

def get_recent_fixtures(json_file_path: str, num_fixtures: int = 10) -> dict:
    """
    Extrae los N fixtures más recientes ordenados por fecha
    
    Args:
        json_file_path: Ruta al archivo JSON
        num_fixtures: Número de fixtures a extraer (default 10)
    
    Returns:
        Diccionario con los fixtures más recientes
    """
    
    # Cargar JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Convertir a lista y ordenar por fecha
    fixtures_list = []
    for fixture_id, fixture_data in data.items():
        fixture_data['fixture_id'] = fixture_id
        fixtures_list.append(fixture_data)
    
    # Ordenar por fecha descendente (más recientes primero)
    fixtures_list.sort(
        key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'),
        reverse=True
    )
    
    # Tomar los N más recientes
    recent_fixtures = fixtures_list[:num_fixtures]
    
    # Convertir de vuelta a diccionario
    result = {}
    for fixture in recent_fixtures:
        fixture_id = fixture.pop('fixture_id')
        result[fixture_id] = fixture
    
    return result


def append_team_stat(team_id, fixture_id, team_side, fixture, stats, path_dir):
    """Save team stats grouped per team and per fixture."""
    path = os.path.join(path_dir, f"{team_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    league_info = fixture["league"]
    game_info = fixture["fixture"]
    goals = fixture["goals"]

    existing[str(fixture_id)] = {
        "league": league_info["name"],
        "season": league_info["season"],
        "date": game_info["date"].split("T")[0],
        "home": fixture["teams"]["home"]["id"],
        "away": fixture["teams"]["away"]["id"],
        "side": team_side,
        "stats": {s["type"]: s["value"] for s in stats if s["value"] is not None},
        "goals": {
            "for": goals["home"] if team_side == "home" else goals["away"],
            "against": goals["away"] if team_side == "home" else goals["home"],
        },
    }

    # save_json(path, existing)

    return existing


def load_transform_fixture_data_per_date_range(start_date, end_date) -> pd.DataFrame:
    """
    Processes fixtures already saved under data/fixtures/YYYY-MM-DD/*.json
    For each fixture, extracts stats, predictions, and odds.
    """
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    complete_fixture_data = []
    Y = []

    while current_date <= end_date:

        

        date_str = current_date.strftime("%Y-%m-%d")
        date_dir = os.path.join(FIXTURES_DIR, date_str)
        print(f"\n📅 Processing fixtures for {date_str}...")

        if not os.path.exists(date_dir):
            print(f"⚠️ No local fixture folder for {date_str}, skipping.")
            current_date += timedelta(days=1)
            continue

        fixture_files = [f for f in os.listdir(date_dir) if f.endswith(".json")]
        if not fixture_files:
            print(f"⚠️ No fixtures in {date_dir}")
            current_date += timedelta(days=1)
            continue

        print(f"Found {len(fixture_files)} fixtures for {date_str}")

        # Process each fixture
        for fix_file in fixture_files:

            fixture_data = {}

            fixture_path = os.path.join(date_dir, fix_file)
            fixture = load_json(fixture_path)
            if not fixture:
                continue

            # fixture_data["fixture"] = fixture

            fixture_id = fixture["fixture"]["id"]
            fixture_date = fixture["fixture"]["date"] #timezone UTC
            home_team_id = fixture["teams"]["home"]["id"]
            home_team_name = fixture["teams"]["home"]["name"]
            
            away_team_id = fixture["teams"]["away"]["id"]
            away_team_name = fixture["teams"]["away"]["name"]

            teams = [home_team_id, away_team_id]

    
            # Team statistics
            print(f"📈 Fetching stats for fixture {fixture_id}")
            historic_stats = {}
            for team_id in teams:
                stats = os.path.join(TEAM_STATS_DIR, f"{team_id}.json")
                if not os.path.exists(stats):
                    print(f"⚠️ No stats data for team {team_id} in fixture {fixture_id}")
                    print(stats)
                    #print current directory
                    print(os.getcwd())
                else:   

                    recent_fixtures_stats, y = get_fixtures_stats_target(stats, str(fixture_id), num_fixtures=5)
                    # team_id = stats["team"]["id"]
                    print(f"Team {team_id} stats from fixture {fixture_id}")
                    print(recent_fixtures_stats)

                    if recent_fixtures_stats == None or y == None:
                        print(f"⚠️ No recent fixtures stats found for team {team_id} before fixture {fixture_id}")
                        continue

                    team_location = "home" if team_id == fixture["teams"]["home"]["id"] else "away"

                    home_bin = True if team_location == "home" else False

                    goals = []
                    shots_on_goal = []
                    pass_accuracy = []
                    possession = []
                    pass_accuracy = []
                    yellow_cards = []
                    red_cards = []
                    corners = []
                    fouls = []
                    streak = []

                    y_data = {
                        "team_id": team_id,
                        'fixture_id': fixture_id,
                        'corners': y[0]['stats'].get("Corner Kicks", 0) if y else 0
                    }
                    

                    for fixture_stats in recent_fixtures_stats:
                        #fixture_stats["stats"]
                        
                        stats = fixture_stats["stats"]
                        goals_data = fixture_stats["goals"]
                        print(goals_data)

                        goals.append(goals_data.get("for") or 0)
                        shots_on_goal.append(stats.get("Shots on Goal", 0))
                        pass_accuracy.append(float(str(stats.get("Passes %", 0)).strip('%'))) #transaformar a float
                        possession.append(float(str(stats.get("Ball Possession", 0)).strip('%'))) #transaformar a float
                        yellow_cards.append(stats.get("Yellow Cards", 0))
                        red_cards.append(stats.get("Red Cards", 0))
                        corners.append(stats.get("Corner Kicks", 0))
                        fouls.append(stats.get("Fouls", 0))

                        #if null then 0
                        goals_for = goals_data.get("for") or 0
                        goals_against = goals_data.get("against") or 0

                        print("GOALS FOR:", goals_for, " - GOALS AGAINST:", goals_against)

                        # Racha (Streak)\
                        if goals_for > goals_against:
                            streak.append("W")
                        elif goals_for < goals_against:
                            streak.append("L")
                        else:
                            streak.append("D")
                    


                    avg_goals = np.mean(goals) if goals else 0
                    avg_shots_on_goal = np.mean(shots_on_goal) if shots_on_goal else 0
                    avg_pass_accuracy = np.mean(pass_accuracy) if pass_accuracy else 0
                    avg_possession = np.mean(possession) if possession else 0
                    avg_yellow_cards = np.mean(yellow_cards) if yellow_cards else 0
                    avg_red_cards = np.mean(red_cards) if red_cards else 0
                    avg_corners = np.mean(corners) if corners else 0
                    avg_fouls = np.mean(fouls) if fouls else 0

                    historic_stats= {
                        "team_id" : team_id,
                        'fixture_id': fixture_id,
                        "team_name": fixture["teams"][team_location]["name"],
                        "avg_goals": round(avg_goals, 2),
                        "shots_on_goal": round(avg_shots_on_goal, 2),
                        "pass_accuracy": round(avg_pass_accuracy, 2),
                        "possession": round(avg_possession, 2),
                        "yellow_cards": round(avg_yellow_cards, 2),
                        "red_cards": round(avg_red_cards, 2),
                        "corners": round(avg_corners, 2),
                        "fouls": round(avg_fouls, 2),
                        "streak": ''.join(streak),
                        "num_matches": len(recent_fixtures_stats),
                        "home_bin" : home_bin
                    }

            # fixture_data["stats"] = historic_stats
                    complete_fixture_data.append(historic_stats)
                    Y.append(y_data)
            
        
        current_date += timedelta(days=1)

    print("\n✅ Pipeline complete.")
    df_final = pd.DataFrame(complete_fixture_data)  
    Y = pd.DataFrame(Y)
    
    return df_final, Y
