import pandas as pd

def load_and_clean(path):
    #  Load CSV
    df = pd.read_csv(path)

    # Keep relevant columns
    df = df[[
        "winner_name", "loser_name", "surface",
        "winner_rank", "loser_rank",
        "w_ace", "l_ace"
    ]]

    # Drop rows with missing ranks
    df = df.dropna(subset=["winner_rank", "loser_rank", "w_ace", "l_ace"])

    # Winner perspective
    winners = pd.DataFrame({
        "player": df["winner_name"],
        "opponent": df["loser_name"],
        "surface": df["surface"],
        "player_rank": df["winner_rank"],
        "opponent_rank": df["loser_rank"],
        "player_ace": df["w_ace"],
        "opponent_ace": df["l_ace"],
        "target": 1
    })

    # Loser perspective
    losers = pd.DataFrame({
        "player": df["loser_name"],
        "opponent": df["winner_name"],
        "surface": df["surface"],
        "player_rank": df["loser_rank"],
        "opponent_rank": df["winner_rank"],
        "player_ace": df["l_ace"],
        "opponent_ace": df["w_ace"],
        "target": 0
    })

    # Combine both
    final_df = pd.concat([winners, losers], ignore_index=True)

    return final_df

if __name__ == "__main__":
    data = load_and_clean("data/atp_matches_2024.csv")
    print(data.head(10))
