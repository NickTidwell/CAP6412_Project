import pandas as pd
import os
from model_info import MODEL_LIST
INIT_VALUE = 500
SCALING_FACTOR = 400
class EloRatingSystem:
    def __init__(self, k_factor=32):
        """
        Initializes the Elo rating system.
        Args:
            k_factor (float, optional): Sensitivity factor (default is 32).
        """
        self.k_factor = k_factor
        self.elo_df = self.load_elo_csv()

    def load_elo_csv(self):
        elo_csv_path = "output/elo.csv"
        if os.path.exists(elo_csv_path):
            return pd.read_csv(elo_csv_path, index_col="Model")
        else:
            model_list = MODEL_LIST
            initial_elo = INIT_VALUE
            elo_data = {"Model": model_list, "ELO": [initial_elo] * len(model_list)}
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

        self.elo_df.to_csv("output/elo.csv")  # Save updated ELO ratings

        return new_rating_a, new_rating_b

    def calculate_expected_score(self, player_a_rating, player_b_rating):
        return 1 / (10 ** ((player_b_rating - player_a_rating) / SCALING_FACTOR) + 1)
    def print_elo(self):
        print(self.elo_df)