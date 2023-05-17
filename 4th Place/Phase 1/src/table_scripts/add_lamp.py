#
# Author: Yudong Lin
#
# obtain the latest forecasts information and add it to the data frame
#

import pandas as pd


# add lamp forecast weather information
def add_lamp(
    now: pd.Timestamp,
    flights_selected: pd.DataFrame,
    data_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    # forecasts that can be used for prediction
    forecasts_available: pd.DataFrame = data_tables["lamp"]
    # numerical features for lamp
    numerical_feature: tuple[str, ...] = (
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
    )
    # categorical features for lamp
    categorical_feature: tuple[str, ...] = ("cloud", "lightning_prob", "precip")
    # if exists forecasts that can be used for prediction
    if forecasts_available.shape[0] > 0:
        # the latest forecast
        latest_forecast: pd.DataFrame | None = None
        # counter to monitoring hours going forward
        hour_f: int = 0
        # when no valid forecast is found
        while latest_forecast is None:
            # round time to the nearest hour
            forecast_timestamp_to_look_up: pd.Timestamp = (
                now - pd.Timedelta(hours=hour_f)
            ).round("H")
            # select all rows contain this forecast_timestamp
            forecasts: pd.DataFrame = forecasts_available.loc[
                forecasts_available.forecast_timestamp == forecast_timestamp_to_look_up
            ]
            # get the latest forecast
            latest_forecast = (
                forecasts.iloc[forecasts.index.get_indexer([now], method="nearest")]
                if forecasts.shape[0] > 0
                else None
            )
            # if a valid latest forecast is found
            if latest_forecast is not None:
                # then update value
                for key in numerical_feature + categorical_feature:
                    flights_selected[key] = latest_forecast[key].values[0]
                # and break the loop
                return flights_selected
            # if no forecast within 30 hours can be found
            elif hour_f > 30:
                break
            hour_f += 1
    for key in numerical_feature:
        flights_selected[key] = 0
    for key in categorical_feature:
        flights_selected[key] = "UNK"
    return flights_selected
