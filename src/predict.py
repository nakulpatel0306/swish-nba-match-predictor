# Auto-generated from notebook: predict.ipynb

# ---- Cell ----
import pandas as pd

# ---- Cell ----
df = pd.read_csv("nba_games.csv",index_col=0)

# ---- Cell ----
df

# ---- Cell ----
df = df.sort_values("date")

# ---- Cell ----
df = df.reset_index(drop=True)

# ---- Cell ----
df

# ---- Cell ----
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

# ---- Cell ----
df

# ---- Cell ----
def add_target(Team):
    Team["target"] = Team["won"].shift(-1)
    return Team

df = df.groupby("Team", group_keys=False).apply(add_target)

# ---- Cell ----
df[df["Team"] == "WAS"]

# ---- Cell ----
df["target"][pd.isnull(df["target"])] = 2

# ---- Cell ----
df["target"] = df["target"].astype(int, errors="ignore")

# ---- Cell ----
df

# ---- Cell ----
df["won"].value_counts()

# ---- Cell ----
df["target"].value_counts()

# ---- Cell ----
nulls = pd.isnull(df)

# ---- Cell ----
nulls = nulls.sum()

# ---- Cell ----
nulls = nulls[nulls > 0]

# ---- Cell ----
valid_columns = df.columns[~df.columns.isin(nulls.index)]

# ---- Cell ----
valid_columns

# ---- Cell ----
df = df[valid_columns].copy()

# ---- Cell ----
df

# ---- Cell ----
%pip install scikit-learn

# ---- Cell ----
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)

sfs=SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward",cv=split)

# ---- Cell ----
removed_columns = ["season","date","won","target","Team","Team_opp"]

# ---- Cell ----
selected_columns = df.columns[~df.columns.isin(removed_columns)]

# ---- Cell ----
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])

# ---- Cell ----
df

# ---- Cell ----
sfs.fit(df[selected_columns], df["target"])

# ---- Cell ----
predictors = list(selected_columns[sfs.get_support()])

# ---- Cell ----
predictors

# ---- Cell ----
def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []

    seasons = sorted(data["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]

        train = data[data["season"] < season]
        test = data[data["season"] == season]

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)

        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "predictions"]

        all_predictions.append(combined)
    return pd.concat(all_predictions)

# ---- Cell ----
predictions = backtest(df,rr,predictors)

# ---- Cell ----
predictions

# ---- Cell ----
from sklearn.metrics import accuracy_score

predictions = predictions[predictions["actual"] != 2]
accuracy_score(predictions["actual"], predictions["predictions"])

# ---- Cell ----
df.groupby("Home").apply(lambda x: x[x["won"] == 1].shape[0]/x.shape[0])

# ---- Cell ----
df

# ---- Cell ----
df_rolling = df[list(selected_columns) + ["won", "Team", "season"]]

# ---- Cell ----
df_rolling

# ---- Cell ----
def find_team_averages(Team):
    Team = Team.select_dtypes(include=[float, int])
    rolling = Team.rolling(10).mean()
    return rolling

df_rolling = df_rolling.groupby(["Team","season"], group_keys=False).apply(find_team_averages)

# ---- Cell ----
df_rolling

# ---- Cell ----
rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols

df = pd.concat([df,df_rolling], axis=1)

# ---- Cell ----
df

# ---- Cell ----
df = df.dropna()

# ---- Cell ----
df

# ---- Cell ----
def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("Team", group_keys=False).apply(lambda x: shift_col(x, col_name))

df["home_next"] = add_col(df, "Home")
df["Team_opp_next"] = add_col(df, "Team_opp")
df["date_next"] = add_col(df, "date")

# ---- Cell ----
df

# ---- Cell ----
df = df.copy()

# ---- Cell ----
full = df.merge(df[rolling_cols + ["Team_opp_next", "date_next", "Team"]], 
                left_on=["Team", "date_next"], 
                right_on=["Team_opp_next", "date_next"]
               )

# ---- Cell ----
full

# ---- Cell ----
full[["Team_x", "Team_opp_next_x", "Team_y","Team_opp_next_y", "date_next"]]

# ---- Cell ----
removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns

# ---- Cell ----
removed_columns

# ---- Cell ----
selected_columns = full.columns[~full.columns.isin(removed_columns)]

# ---- Cell ----
sfs.fit(full[selected_columns], full["target"])

# ---- Cell ----
predictors = list(selected_columns[sfs.get_support()])

# ---- Cell ----
predictors

# ---- Cell ----
predictions = backtest(full,rr,predictors)

# ---- Cell ----
accuracy_score(predictions["actual"], predictions["predictions"])
