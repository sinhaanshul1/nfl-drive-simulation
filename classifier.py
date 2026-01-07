
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TEST_SIZE = 0.2
EPOCHS = 20

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df['is_short_yardage'] = (df['ydstogo'] <= 3).astype(int)
    df['is_long_yardage'] = (df['ydstogo'] >= 7).astype(int)
    df['is_third_down'] = (df['down'] == 3).astype(int)
    df['is_fourth_down'] = (df['down'] == 4).astype(int)
    df['in_red_zone'] = (df['yardline_100'] <= 20).astype(int)
    df['in_scoring_position'] = (df['yardline_100'] <= 10).astype(int)
    df['backed_up'] = (df['yardline_100'] >= 80).astype(int)
    df['is_trailing'] = (df['score_differential'] < 0).astype(int)
    df['is_leading_big'] = (df['score_differential'] >= 14).astype(int)
    df['is_close_game'] = (df['score_differential'].abs() <= 7).astype(int)
    df['late_in_half'] = ((df['quarter_seconds_remaining'] <= 300) & (df['qtr'].isin([2, 4]))).astype(int)
    df['very_late_game'] = ((df['quarter_seconds_remaining'] <= 120) & (df['qtr'].isin([2, 4]))).astype(int)
    df['trailing_late'] = ((df['is_trailing'] == 1) & (df['quarter_seconds_remaining'] <= 300) & (df['qtr'].isin([2, 4]))).astype(int)
    df['yards_per_down'] = df['ydstogo'] / (5 - df['down'])
    return df

def load_data(years: list[str], team: str):
    pbp = nfl.import_pbp_data(years)
    pbp = pbp.loc[:, ~pbp.columns.duplicated()]



    

    FEATURES = [
        'play_id', 'game_id', 'home_team', 'away_team', 'season_type', 'week', 'posteam', 'defteam', 'yardline_100', 'game_date', 'quarter_seconds_remaining',
        'qtr', 'down', 'ydstogo', 'drive', 'goal_to_go', 'desc', 'play_type', 'yards_gained', 'pass_length', 'pass_location', 'run_location', 'run_gap', 
        'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'score_differential', 'posteam_score', 'defteam_score'
    ]
    TEST_FEATURES = [
        'posteam', 'yardline_100', 'quarter_seconds_remaining', 'qtr', 'down', 'ydstogo', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'score_differential'
    ]
    ENGINEERED_FEATURES = [
        'is_short_yardage', 'is_long_yardage', 'is_third_down', 'is_fourth_down',
        'in_red_zone', 'in_scoring_position', 'backed_up',
        'is_trailing', 'is_leading_big', 'is_close_game',
        'late_in_half', 'very_late_game', 'trailing_late',
        'yards_per_down'
    ]
    

    
    LABEL = "play_type"
    pbp = add_engineered_features(pbp)

    # select only features that exist in the dataframe to avoid KeyError
    selected_features = [f for f in TEST_FEATURES + ENGINEERED_FEATURES if f in pbp.columns]
    missing_features = [f for f in TEST_FEATURES + ENGINEERED_FEATURES if f not in pbp.columns]
    if missing_features:
        print("Ignoring missing features:", missing_features)

    pbp = pbp.loc[
        pbp['play_type'].isin(['pass', 'run']),
        selected_features + ENGINEERED_FEATURES + [LABEL]
    ]

    print(pbp.head())



    team_data = pbp.loc[pbp['posteam'] == team, TEST_FEATURES + ENGINEERED_FEATURES + [LABEL]]
    team_data['play_type_encoded'] = team_data['play_type'].map({'run': 0, 'pass': 1})
    team_data = team_data.dropna()
    # cols_to_exclude = ['B', 'D']
    columns_to_scale = ['quarter_seconds_remaining', 'yardline_100', 
    'score_differential', 'down', 'qtr',
    'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
    'yards_per_down']

    # Create a copy
    team_data_scaled = team_data.copy()

    # Scale only specific columns
    scaler = StandardScaler()
    team_data_scaled[columns_to_scale] = scaler.fit_transform(team_data[columns_to_scale])
    return team_data_scaled.drop(columns=['posteam', 'play_type', 'play_type_encoded']).values, team_data['play_type_encoded'].values



def main():


    # Get image arrays and labels for all image files
    information, play_type_labels = load_data([2021, 2024], "PIT")


    

    # Split data into training and testing sets
    # play_type_labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(information), np.array(play_type_labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model(input_size=x_train.shape[1], output_shape=2)

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    model.evaluate(x_test,  y_test, verbose=2)
    print(np.isnan(information).sum())   # should be 0
    print(np.isinf(information).sum())


def get_model(input_size: int, output_shape: int):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(input_size,)),
        
        # Wider, shallower network with proper regularization
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss='binary_crossentropy',
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    main()







