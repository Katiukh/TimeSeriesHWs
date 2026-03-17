import numpy as np
import pandas as pd

from .metrics import evaluate_forecasts
from .models import CatBoostDirectMIMO


def make_holdout_split(
    data: pd.DataFrame,
    id_col: str = "unique_id",
    timestamp_col: str = "ds",
    value_col: str = "y",
    history: int = 24,
    horizon: int = 18,
):
    data = data.sort_values([id_col, timestamp_col]).reset_index(drop=True)

    min_len = history + horizon
    lengths = data.groupby(id_col).size()
    valid_ids = lengths[lengths >= min_len].index

    data = data[data[id_col].isin(valid_ids)].copy()
    data = data.sort_values([id_col, timestamp_col]).reset_index(drop=True)

    train_parts = []
    holdout_parts = []
    truth_parts = []

    for uid, g in data.groupby(id_col, sort=False):
        g = g.sort_values(timestamp_col).reset_index(drop=True)

        train_part = g.iloc[:-horizon].copy()
        holdout_part = g.iloc[-(history + horizon):].copy()
        truth_part = g.iloc[-horizon:][[id_col, timestamp_col, value_col]].copy()

        holdout_part.loc[holdout_part.index[-horizon:], value_col] = np.nan

        train_parts.append(train_part)
        holdout_parts.append(holdout_part)
        truth_parts.append(truth_part)

    train_df = pd.concat(train_parts, ignore_index=True)
    holdout_df = pd.concat(holdout_parts, ignore_index=True)
    truth_df = pd.concat(truth_parts, ignore_index=True)

    return train_df, holdout_df, truth_df


def run_holdout_experiment(
    data: pd.DataFrame,
    feature_config: dict,
    model_horizon: int = 6,
    history: int = 24,
    horizon: int = 18,
    freq: str = "M",
    id_col: str = "unique_id",
    timestamp_col: str = "ds",
    value_col: str = "y",
    seasonality: int = 12,
):
    train_df, holdout_df, truth_df = make_holdout_split(
        data=data,
        id_col=id_col,
        timestamp_col=timestamp_col,
        value_col=value_col,
        history=history,
        horizon=horizon,
    )

    model = CatBoostDirectMIMO(
        model_horizon=model_horizon,
        history=history,
        horizon=horizon,
        freq=freq,
        feature_config=feature_config,
    )

    model.fit(
        train_df,
        holdout_df,
        id_col=id_col,
        timestamp_col=timestamp_col,
        value_col=value_col,
    )

    pred_df = model.predict(
        holdout_df,
        id_col=id_col,
        timestamp_col=timestamp_col,
        value_col=value_col,
    )

    metrics = evaluate_forecasts(
        train_df=train_df,
        truth_df=truth_df,
        pred_df=pred_df,
        id_col=id_col,
        timestamp_col=timestamp_col,
        value_col=value_col,
        seasonality=seasonality,
    )

    return {
        "train_df": train_df,
        "holdout_df": holdout_df,
        "truth_df": truth_df,
        "pred_df": pred_df,
        "merged_predictions": metrics["merged_predictions"],
        "sMAPE": metrics["sMAPE"],
        "MASE": metrics["MASE"],
    }
