import pandas as pd
import json
import os
from model_info import MODEL_LIST
INIT_VALUE = 500
SCALING_FACTOR = 400
class EloRatingSystem:
    def __init__(self, k_factor=32, elo_csv_path = "output/elo.csv"):
        """
        Initializes the Elo rating system.
        Args:
            k_factor (float, optional): Sensitivity factor (default is 32).
        """
        self.k_factor = k_factor
        self.elo_df = self.load_elo_csv(elo_csv_path)
        self.elo_csv_path = elo_csv_path

    def load_elo_csv(self, elo_csv_path):
        elo_csv_path = elo_csv_path
        if os.path.exists(elo_csv_path):
            return pd.read_csv(elo_csv_path, index_col="Model")
        else:
            model_list = MODEL_LIST
            initial_elo = INIT_VALUE
            elo_data = {"Model": model_list, "ELO": [initial_elo] * len(model_list), "TEST_COUNT": 0}
            elo_df = pd.DataFrame(elo_data)
            elo_df.set_index('Model', inplace=True)
            elo_df.to_csv(elo_csv_path, index=False)
            return elo_df
            
    def update_ratings(self, player_a, player_b, score):
        """
        Args:
            score (float): Match outcome (win = 1, draw = 0.5, loss = 0).
        """
        player_a_rating = self.elo_df.loc[player_a, "ELO"]
        player_b_rating = self.elo_df.loc[player_b, "ELO"]

        expected_a = self.calculate_expected_score(player_a_rating, player_b_rating)
        expected_b = 1 - expected_a

        elo_change_a = self.k_factor * (score - expected_a)
        elo_change_b = self.k_factor * (1 - score - expected_b)

        new_rating_a = player_a_rating + elo_change_a
        new_rating_b = player_b_rating + elo_change_b

        self.elo_df.loc[player_a, "ELO"] = new_rating_a
        self.elo_df.loc[player_b, "ELO"] = new_rating_b

        self.elo_df.loc[player_a, "TEST_COUNT"] += 1
        self.elo_df.loc[player_b, "TEST_COUNT"] += 1
        self.elo_df.to_csv(self.elo_csv_path)  # Save updated ELO ratings

        return new_rating_a, new_rating_b

    def calculate_expected_score(self, player_a_rating, player_b_rating):
        return 1 / (10 ** ((player_b_rating - player_a_rating) / SCALING_FACTOR) + 1)
        
    def print_elo(self):
        print(self.elo_df)


class StatsCounter:
    def __init__(self, json_file):
        self.json_file = json_file
        self.dictionary = self.load_from_json()

    def update_count(self, string1, string2):
        # Sort the strings to ensure consistent ordering
        sorted_strings = tuple(sorted([string1, string2]))
        key = sorted_strings[0] + "#" + sorted_strings[1]
        if key in self.dictionary:
            self.dictionary[key] += 1
        else:
            self.dictionary[key] = 1
        self.save_to_json()

    def print_dictionary(self):
        for key, value in sorted(self.dictionary.items()):
            parts = key.split("#")

            print(f'{parts[0]} vs {parts[1]}: {value}')

    def save_to_json(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.dictionary, f)

    def load_from_json(self):
        if not os.path.exists(self.json_file):
            with open(self.json_file, 'w') as f:
                f.write('{}')
        try:
            with open(self.json_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
