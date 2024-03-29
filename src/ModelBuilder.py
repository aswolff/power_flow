import pandas as pd
import sys
import numpy as np
from config import *
from lightgbm import LGBMRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV
from skforecast.ForecasterBaseline import ForecasterEquivalentDate
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
import plotly.graph_objs as go
from plotly.offline import plot

END_TRAIN = timedelta(days=690, minutes=1)
END_VALIDATION = timedelta(days=55, minutes=1)

class ModelBuilder:
    def __init__(self):
        try:
            self.data = pd.read_csv(DATA_PATH, parse_dates=['time'], index_col='time')
        except FileNotFoundError:
            print(f"The file {DATA_PATH} does not exist. Please create data with DataProcessor")
            sys.exit(1)
        self.end_train = self.data.index.max() - END_TRAIN
        self.end_validation = self.data.index.max() - END_VALIDATION
        self.train, self.validation, self.test = self.create_data_splits()
        #self.forecaster = self.tune_hyperparameters()
        
    def create_data_splits(self):
        data_train = self.data.loc[: self.end_train, :].copy()
        data_val = self.data.loc[self.end_train: self.end_validation, :].copy()
        data_test = self.data.loc[self.end_validation:, :].copy()
        print(" Created Train, Validation and Test Split")
        print(f' Train Dates: {data_train.index.min()} --- {data_train.index.max()} (n={len(data_train)})')
        print(f' Validation Dates: {data_val.index.min()} --- {data_val.index.max()} (n={len(data_val)})')
        print(f' Test Dates: {data_test.index.min()} --- {data_test.index.max()} (n={len(data_test)})')
        return data_train, data_val, data_test
    
    def create_baseline(self):
        """"
        Creates simple baseline model
        Predicts that the value of the current hour will be the same as the one 24 hours prior
        """
        forecaster = forecaster = ForecasterEquivalentDate(
                 offset    = pd.DateOffset(days=1),
                 n_offsets = 1
             )

        # Train forecaster
        # ==============================================================================
        forecaster.fit(y=self.data.loc[:self.end_validation, 'value'])
        
        self.backtest_forecaster(forecaster)


    def create_multi_step(self):
        """
        Creates recursive multi-step forecast model
        Uses the ForecasterAutoreg from skforecast to create 24 hours of lag values to predict demand
        """

        forecaster = ForecasterAutoreg(
                 regressor = LGBMRegressor(random_state=16000, verbose=-1),
                 lags      = 24
             )

        forecaster.fit(y=self.data.loc[:self.end_validation, 'value'])

        self.backtest_forecaster(forecaster)

    def tune_hyperparameters(self):
        forecaster = ForecasterAutoreg(
                 regressor = LGBMRegressor(random_state=15926, verbose=-1),
                 lags      = 24, # This value will be replaced in the grid search
             )

        lags_grid = [24, [1, 2, 3, 23, 24, 25, 47, 48, 49]]

        results_search, frozen_trial = bayesian_search_forecaster(
                                   forecaster         = forecaster,
                                   y                  = self.data.loc[:self.end_validation, 'value'],
                                   steps              = 24,
                                   metric             = 'mean_absolute_error',
                                   search_space       = self.search_space,
                                   lags_grid          = lags_grid,
                                   initial_train_size = len(self.data[:self.end_train]),
                                   refit              = False,
                                   n_trials           = 20, # Increase for more exhaustive search
                                   random_state       = 123,
                                   return_best        = True,
                                   n_jobs             = 'auto',
                                   verbose            = True,
                                   show_progress      = True
                               )
        
        self.backtest_forecaster(forecaster)
        return forecaster
    
    def backtest_forecaster(self, forecaster):
        metric, predictions = backtesting_forecaster(
                          forecaster         = forecaster,
                          y                  = self.data['value'],
                          steps              = 24,
                          metric             = 'mean_absolute_error',
                          initial_train_size = len(self.data[:self.end_validation]),
                          refit              = False,
                          n_jobs             = 'auto',
                          verbose            = True,
                          show_progress      = True
                      )

        print(f"Backtest error: {metric:.2f}")
        return metric, predictions


    def search_space(self, trial):
        """
        Regressor hyperparameters search space
        """
        search_space  = {
            'n_estimators'  : trial.suggest_int('n_estimators', 600, 1200, step=100),
            'max_depth'     : trial.suggest_int('max_depth', 3, 10, step=1),
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.5),
            'reg_alpha'     : trial.suggest_float('reg_alpha', 0, 1, step=0.1),
            'reg_lambda'    : trial.suggest_float('reg_lambda', 0, 1, step=0.1),
        } 
        return search_space
    
    def plot_predictions(self, forecaster):
        _, predictions = self.backtest_forecaster(forecaster)
        data_test = self.data.loc[END_VALIDATION:, :].copy()

        trace1 = go.Scatter(x=data_test.index, y=data_test['value'], name="Test Data", mode="lines", line=dict(color='royalblue'))
        trace2 = go.Scatter(x=predictions.index, y=predictions['pred'], name="Predictions", mode="lines", line=dict(color='tomato'))

        fig = go.Figure()
        fig.add_trace(trace1)
        fig.add_trace(trace2)

        fig.update_layout(
            title="Real Value vs Predicted in Test Data",
            xaxis_title="Date Time",
            yaxis_title="Demand",
            width=800,
            height=400,
            margin=dict(l=20, r=20, t=35, b=20),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.1,
                xanchor="left",
                x=0.001
            )
        )

        # Generate and open the plot in a web browser
        plot(fig, filename='./Plots/demand_predicted.html', auto_open=True)

    def create_poly_features(self):
        exogenous_features = pd.read_csv(EXO_PATH)

        transformer_poly = PolynomialFeatures(
                       degree           = 2,
                       interaction_only = True,
                       include_bias     = False
                   ).set_output(transform="pandas")

        poly_cols = [
            'month_sin',
            'month_cos',
            'week_of_year_sin',
            'week_of_year_cos',
            'week_day_sin',
            'week_day_cos',
            'hour_day_sin',
            'hour_day_cos',
            'sunrise_hour_sin',
            'sunrise_hour_cos',
            'sunset_hour_sin',
            'sunset_hour_cos',
            'daylight_hours',
            'is_daylight',
            'temp_roll_mean_1_day',
            'temp_roll_mean_7_day',
            'temp_roll_max_1_day',
            'temp_roll_min_1_day',
            'temp_roll_max_7_day',
            'temp_roll_min_7_day',
            'temperature'
        ]

        poly_features = transformer_poly.fit_transform(exogenous_features[poly_cols].dropna())
        poly_features = poly_features.drop(columns=poly_cols)
        poly_features.columns = [f"poly_{col}" for col in poly_features.columns]
        poly_features.columns = poly_features.columns.str.replace(" ", "__")
        exogenous_features = pd.concat([exogenous_features, poly_features], axis=1)
        return exogenous_features