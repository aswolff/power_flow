import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import calendar
import sys
from config import *

class DataExplorer:
    def __init__(self):
        try:
            self.data = pd.read_csv(DATA_PATH, parse_dates=['time'], index_col='time')
        except FileNotFoundError:
            print(f"The file {DATA_PATH} does not exist. Please create data with DataProcessor")
            sys.exit(1)

    def plot_data(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['value'], mode='lines', name='Data'))
        fig.update_layout(
            title='Hourly energy demand',
            xaxis_title="Time",
            yaxis_title="Demand",
            legend_title="Partition:",
            width=850,
            height=400,
            margin=dict(l=20, r=20, t=35, b=20),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0.001
            )
        )
        # Create and open an HTML file with the plot
        plot(fig, filename='./Plots/demand.html', auto_open=True)

    def plot_zoom(self, start_days=1127, end_days=1158):
        # Calculate the zoom start and end times
        zoom_start = self.data.index.min() + pd.Timedelta(days=start_days)
        zoom_end = self.data.index.min() + pd.Timedelta(days=end_days)

        # Create the zoomed plot
        zoomed_fig = go.Figure()
        zoomed_fig.add_trace(go.Scatter(
            x=self.data.loc[zoom_start:zoom_end].index, 
            y=self.data.loc[zoom_start:zoom_end]['value'], 
            mode='lines', 
            name='Zoomed Data',
            line=dict(color='blue', width=1)
        ))

        # Add title and adjust layout for the zoomed plot
        zoomed_fig.update_layout(
            title=f'Zoomed Electricity demand: {zoom_start.strftime("%Y-%m-%d")} to {zoom_end.strftime("%Y-%m-%d")}',
            xaxis_title="Time",
            yaxis_title="Demand",
            showlegend=False,
            width=850,
            height=400
        )

        # Create and open two HTML files with the plots
        plot(zoomed_fig, filename='./Plots/demand_zoomed.html', auto_open=True)

    def plot_monthly(self):
        # Create a new DataFrame with the 'month' column
        data_with_month = self.data.copy()
        data_with_month['month'] = data_with_month.index.month

        # Aggregate the data by month
        monthly_data = data_with_month.groupby('month')['value'].apply(list).reset_index()

        # Create the boxplot traces
        traces = []
        for i, month in monthly_data.iterrows():
            traces.append(go.Box(
                y=month['value'],
                name=calendar.month_abbr[month['month']],
                boxpoints='all',
                jitter=0.5,
                whiskerwidth=0.2,
                marker_size=2,
                line_width=1
            ))

        # Calculate the median values for the line plot
        medians = data_with_month.groupby('month')['value'].median().sort_index()

        # Create the median trace
        median_trace = go.Scatter(
            x=list(calendar.month_abbr[1:]),
            y=medians,
            mode='lines+markers',
            name='Median',
            line=dict(color='orange', width=0.8)
        )

        
        fig = go.Figure(traces)
        fig.add_trace(median_trace)

        
        fig.update_layout(
            title='Demand distribution by month',
            xaxis=dict(title='Month'),
            yaxis=dict(title='Demand'),
            showlegend=False,
            width=660,
            height=300
        )

        plot(fig, filename='./Plots/demand_by_month.html', auto_open=True)

    def plot_week_day(self):
        # Create a new DataFrame with the 'week_day' column
        data_with_week = self.data.copy()
        data_with_week['week_day'] = data_with_week.index.day_of_week + 1  # Monday=1, Sunday=7

        # Prepare the data for boxplot - list of values for each week day
        box_data = [data_with_week[data_with_week['week_day'] == day]['value'] for day in range(1, 8)]

        # Create the boxplot traces
        traces = [go.Box(y=values, name=calendar.day_name[day-1], boxpoints='all', jitter=0.5, whiskerwidth=0.2, marker_size=2, line_width=1) for day, values in enumerate(box_data, start=1)]

        # Calculate the median values for the line plot
        medians = data_with_week.groupby('week_day')['value'].median().reindex(range(1, 8))

        # Create the median trace
        median_trace = go.Scatter(
            x=[calendar.day_name[day-1] for day in range(1, 8)],
            y=medians,
            mode='lines+markers',
            name='Median',
            line=dict(color='orange', width=0.8)
        )

        fig = go.Figure(traces)
        fig.add_trace(median_trace)

        fig.update_layout(
            title='Demand distribution by week day',
            xaxis=dict(title='Week Day', type='category'),
            yaxis=dict(title='Demand'),
            showlegend=False,
            width=660,
            height=300
        )
        
        plot(fig, filename='./Plots/demand_by_week_day.html', auto_open=True)

    def plot_hour(self):
        # Create a new DataFrame with the 'hour_day' column
        data_with_hours = self.data.copy()
        data_with_hours['hour_day'] = data_with_hours.index.hour

        # Prepare the data for boxplot - list of values for each hour
        box_data = [data_with_hours[data_with_hours['hour_day'] == hour]['value'] for hour in range(24)]

        # Create the boxplot traces
        traces = [go.Box(y=values, name=str(hour), boxpoints='all', jitter=0.5, whiskerwidth=0.2, marker_size=2, line_width=1) for hour, values in enumerate(box_data)]

        # Calculate the median values for the line plot
        medians = data_with_hours.groupby('hour_day')['value'].median().reindex(range(24))

        # Create the median trace
        median_trace = go.Scatter(
            x=list(range(24)),
            y=medians,
            mode='lines+markers',
            name='Median',
            line=dict(color='orange', width=0.8)
        )

        fig = go.Figure(traces)
        fig.add_trace(median_trace)

        fig.update_layout(
            title='Demand distribution by the hour of the day',
            xaxis=dict(title='Hour of Day', tickmode='array', tickvals=list(range(24)), ticktext=[f"{hour}:00" for hour in range(24)]),
            yaxis=dict(title='Demand'),
            showlegend=False,
            width=660,
            height=300
        )

        plot(fig, filename='./Plots/demand_by_hour_of_day.html', auto_open=True)