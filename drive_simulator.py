import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ydstogo_range(ydstogo: int) -> range:
    if ydstogo <= 3:
        return range(1, 4)
    if ydstogo <= 7:
        return range(4, 8)
    if ydstogo <= 10:
        return range(8, 11)
    return range(11, 100)

def yardline_range(yardline: int) -> range:
    if yardline <= 10:
        return range(1, 11)
    if yardline <= 20:
        return range(11, 21)
    if yardline <= 40:
        return range(21, 41)
    if yardline <= 60:
        return range(41, 61)
    if yardline <= 80:
        return range(61, 81)
    if yardline <= 90:
        return range(81, 91)
    return range(91, 101)

def quarter_seconds_range(quarter_seconds: int, quarter: int) -> range:
    if quarter in (2, 4):
        if quarter_seconds <= 60:
            return range(0, 61)
        if quarter_seconds <= 120:
            return range(61, 121)
        if quarter_seconds <= 300:
            return range(121, 301)
        return range(301, 901)
    else:
        if quarter_seconds <= 300:
            return range(0, 301)
        return range(301, 901)
def score_diff_range(score_diff: int) -> range:
    if score_diff < -14:
        return range(-100, -13)
    if score_diff < -4:
        return range(-13, -3)
    if score_diff < 0:
        return range(-3, 0)
    if score_diff == 0:
        return range(0, 1)
    if score_diff <= 3:
        return range(1, 4)
    if score_diff <= 13:
        return range(4, 14)
    return range(14, 100)





def get_matching_rows(pbp: pd.DataFrame, down: int, ydstogo: int, yardline_100: int, qtr: int, quarter_seconds_remaining: int, score_diff: int) -> pd.DataFrame:
    filtered_df = pbp[
        (pbp['down'] == down) &
        (pbp['ydstogo'].isin(ydstogo_range(ydstogo))) &
        (pbp['yardline_100'].isin(yardline_range(yardline_100))) &
        (pbp['qtr'] == qtr) &
        (pbp['quarter_seconds_remaining'].isin(quarter_seconds_range(quarter_seconds_remaining, qtr))) &
        (pbp['score_differential'].isin(score_diff_range(score_diff)))
    ]
    interesting_columns = [
        'play_id', 'game_id', 'posteam', 'defteam', 'yardline_100', 'quarter_seconds_remaining',
        'qtr', 'down', 'ydstogo', 'play_type', 'yards_gained', 'touchdown', 'fumble', 'interception', 'field_goal_result'
    ]
    return filtered_df[interesting_columns]


def simulate_drive(pbp: pd.DataFrame, down: int, ydstogo: int, yardline_100: int, qtr: int, quarter_seconds_remaining: int, score_diff: int):
    current_state = {
        'down': down,
        'ydstogo': ydstogo,
        'yardline_100': yardline_100,
        'qtr': qtr,
        'quarter_seconds_remaining': quarter_seconds_remaining
    }
    current_state_row = get_matching_rows(pbp, down, ydstogo, yardline_100, qtr, quarter_seconds_remaining, score_diff).sample(n=1)
    hit_terminal_state = False
    plays = 0
    while not hit_terminal_state:
        yards_gained = current_state_row['yards_gained'].values[0]
        plays += 1
        print(f"{int(current_state_row['down'].values[0])}&{int(current_state_row['ydstogo'].values[0])}\tPlay: {current_state_row['play_type'].values[0]}, Yards Gained: {yards_gained}")
        if current_state_row['touchdown'].values[0] == 1:
            print("Touchdown!")
            hit_terminal_state = True
            return 'touchdown', plays
            continue
        if current_state_row['fumble'].values[0] == 1:
            print("Fumble! Drive ends.")
            hit_terminal_state = True
            return 'fumble', plays
            continue
        if current_state_row['interception'].values[0] == 1:
            print("Interception! Drive ends.")
            hit_terminal_state = True
            return 'interception', plays
            continue
        if current_state_row['field_goal_result'].values[0] in ['made', 'missed']:
            print(f"Field Goal {current_state_row['field_goal_result'].values[0]}! Drive ends.")
            hit_terminal_state = True
            return 'made_field_goal' if current_state_row['field_goal_result'].values[0] == 'made' else 'missed_field_goal', plays
            continue
        
        if yards_gained >= current_state['ydstogo']:
            current_state['down'] = 1
            current_state['ydstogo'] = 10
        else:
            current_state['down'] += 1
            current_state['ydstogo'] -= yards_gained
            if current_state['down'] > 4:
                return 'turnover_on_downs', plays
                print("Turnover on downs! Drive ends.")
                hit_terminal_state = True
                continue
        
        current_state['yardline_100'] -= yards_gained
        current_state['quarter_seconds_remaining'] -= np.random.randint(5, 20)
        if current_state['quarter_seconds_remaining'] <= 0:
            if current_state['qtr'] == 2 or current_state['qtr'] == 4:
                hit_terminal_state = True
                print("End of half/game. Drive ends.")
                return 'end_of_half/game', plays
            else:
                current_state['qtr'] += 1
                current_state['quarter_seconds_remaining'] = 900
        
        current_state_row = get_matching_rows(
            pbp,
            current_state['down'],
            current_state['ydstogo'],
            current_state['yardline_100'],
            current_state['qtr'],
            current_state['quarter_seconds_remaining'],
            score_diff
        ).sample(n=1)

    print("Drive simulation complete.")

def print_probs(probs: dict):
    
    total = 0
    for value in probs.values():
        total += value
    for k in probs:
        print(f"{k}: {probs[k]/total:.2%}")

def get_input():
    num_sims = int(input("Enter number of simulations to run: "))
    down = int(input("Enter current down (1-4): "))
    ydstogo = int(input("Enter yards to go: "))
    yardline_100 = int(input("Enter yardline (100 = own goal line, 0 = opponent end zone): "))
    qtr = int(input("Enter current quarter (1-4): "))
    quarter_time = (input("Enter time remaining in quarter: "))
    quarter_seconds = quarter_time.split(':')
    quarter_seconds = int(quarter_seconds[0]) * 60 + int(quarter_seconds[1])
    score_diff = int(input("Enter current score differential (positive if leading, negative if trailing): "))
    return num_sims, down, ydstogo, yardline_100, qtr, quarter_seconds, score_diff

if __name__ == "__main__":
    pbp = nfl.import_pbp_data(list(range(2015, 2025)))
    pbp = pbp.loc[:, ~pbp.columns.duplicated()]
    probs = {'touchdown': 0, 'fumble': 0, 'interception': 0, 'made_field_goal': 0, 'missed_field_goal': 0, 'turnover_on_downs': 0, 'end_of_half/game': 0}
    num_sims, down, ydstogo, yardline_100, qtr, quarter_seconds, score_diff = get_input()
    # print(get_matching_rows(pbp, down=4, ydstogo=8, yardline_100=20, qtr=2, quarter_seconds_remaining=600))
    plays = 0
    touchdown_percent = []
    for k in range(num_sims):
        print('Drive Simulation:', k+1)
        try:
            terminal_state, play_count = simulate_drive(pbp, down=down, ydstogo=ydstogo, yardline_100=yardline_100, qtr=qtr, quarter_seconds_remaining=quarter_seconds, score_diff=score_diff)
            probs[terminal_state] += 1
            plays += play_count
            touchdown_percent.append(probs['touchdown'] / (k+1))
        except ValueError:
            print("No matching rows found for the current state. Ending drive simulation.")
    print_probs(probs)
    print(f"Average plays per drive: {plays/num_sims:.2f}")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, sum(probs.values()) + 1), touchdown_percent)
    plt.xlabel('Number of Simulations')
    plt.ylabel('Touchdown Percentage')
    plt.title('Touchdown Percentage Over Simulations')
    plt.grid()
    plt.show()
        