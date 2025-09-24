import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from data_prep import load_and_clean

# def predict_win_probability(model, rank_diff, ace_diff):
#     # build DataFrame with proper column names

#     X_new = pd.DataFrame({
#         "rank_diff": [rank_diff],
#         "ace_diff": [ace_diff]
#     })

#     prob = model.predict_proba(X_new)[0, 1]
#     return prob

# Load cleaned data
df = load_and_clean("data/atp_matches_2024.csv")



# Feature engineering: ranking difference
df["rank_diff"] = df["opponent_rank"] - df["player_rank"]
df["ace_diff"] = df["player_ace"] - df["opponent_ace"]

X = df[["rank_diff", "ace_diff"]]   # features
y = df["target"]        # target: 1 = win, 0 = loss

print(X.shape, y.shape)  # (n_samples, 2) and (n_samples,)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# p1 = predict_win_probability(model, rank_diff=num, ace_diff=num)
# print(f"Player ranked much better + 3 more aces: {p1:.2%} chance to win")

# p2 = predict_win_probability(model, rank_diff=-20, ace_diff=-5)
# print(f"Player ranked worse + 5 fewer aces: {p2:.2%} chance to win")



## Matplotlib for analyzing

# rank_range = np.linspace(-1000, 1000, 2000).reshape(-1, 1)
# ace_zero = np.zeros_like(rank_range)  # keep ace_diff=0 for simplicity
# probs = model.predict_proba(np.c_[rank_range, ace_zero])[:, 1]

# plt.figure(figsize=(8,6))
# plt.plot(rank_range, probs, color="blue")
# plt.axvline(0, color="red", linestyle="--", label="Equal Rank")
# plt.title("Win Probability vs Rank Difference (Ace Diff = 0)")
# plt.xlabel("Rank Difference (Opponent Rank - Player Rank)")
# plt.ylabel("Win Probability")
# plt.legend()
# plt.grid(True)
# plt.show()




# plt.figure(figsize=(8,6))
# plt.scatter(df["ace_diff"], df["target"], alpha=0.1, s=10, color="green")
# plt.title("Match Outcome vs Ace Difference")
# plt.xlabel("Ace Difference (Player - Opponent)")
# plt.ylabel("Win (1) or Loss (0)")
# plt.show()

