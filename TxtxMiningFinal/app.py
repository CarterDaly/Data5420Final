import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Load data
csv_file = "results/pred_vs_actual.csv"  # Adjust the path to your CSV
df = pd.read_csv(csv_file)

# Convert pubDate to datetime
df['pubDate'] = pd.to_datetime(df['pubDate'])

# Extract year, month, and day (ignoring time)
df['date_only'] = df['pubDate'].dt.date

# Sort the dataframe by pubDate in ascending order
df = df.sort_values(by='pubDate').reset_index(drop=True)

# Sidebar: Application Settings
st.sidebar.title("App Settings")
st.sidebar.text("")
st.sidebar.subheader("Model 1 Filter")

# Two separate sliders for start and end dates
start_date = st.sidebar.date_input(
    "Select Start Date:",
    min_value=df['date_only'].min(),
    max_value=df['date_only'].max(),
    value=df['date_only'].min()
)

end_date = st.sidebar.date_input(
    "Select End Date:",
    min_value=df['date_only'].min(),
    max_value=df['date_only'].max(),
    value=df['date_only'].max()
)

data_view = st.sidebar.selectbox("Select Feature to View:", ["SP500_Close", "Two_Days_Later_Close"])

# Filter data based on the selected start and end dates
filtered_df = df[(df['date_only'] >= start_date) & (df['date_only'] <= end_date)]

# Aggregate and average only the predicted columns (SP500_Close_Predicted and Two_Days_Later_Close_Predicted) based on date_only
agg_df = filtered_df.groupby('date_only').agg({
    'SP500_Close_Predicted': 'mean',
    'Two_Days_Later_Close_Predicted': 'mean'
}).reset_index()

# Main Application Title
st.title("Predicted vs Actual Analysis")
st.text("")
st.markdown("This application leverages financial headline data from the News API to predict stock price movements. It predicts two key outcomes: the stock's closing price on the same day and its closing price two days later, capturing both immediate and short-term price fluctuations.")
st.text("")

# Dynamic Graphs: Predicted vs Actual
st.header(f"{data_view}: Predicted vs Actual (Model 1)")
st.markdown(f"The first model utilizes TF-IDF vectorization of the headlines and the S&P 500's opening price as input features to predict {data_view}.")
fig = go.Figure()

# Add predicted and actual values
fig.add_trace(go.Scatter(
    x=agg_df['date_only'],
    y=agg_df[f"{data_view}_Predicted"],
    mode='lines',
    name='Predicted',
    line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Predicted: %{y}<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=filtered_df['date_only'],
    y=filtered_df[f"{data_view}_Actual"],
    mode='lines',
    name='Actual',
    line=dict(color='green'),
    hovertemplate='Date: %{x}<br>Actual: %{y}<extra></extra>'
))

# Update layout for Predicted vs Actual graph
fig.update_layout(
    title=f"{data_view}: Predicted vs Actual",
    xaxis_title="Date",
    yaxis_title="Value",
    template="plotly_white",
    legend_title="Legend",
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# Error Graph
st.header("Error Analysis")
error_fig = go.Figure()

# Calculate the average error for each date
avg_error_df = filtered_df.groupby('date_only')[f"{data_view}_Error"].mean().reset_index()

# Add average error values as a bar chart
error_fig.add_trace(go.Bar(
    x=avg_error_df['date_only'],
    y=avg_error_df[f"{data_view}_Error"],
    name='Average Error',
    marker_color='red',
    hovertemplate='Date: %{x}<br>Average Error: %{y}<extra></extra>'
))

# Update layout for Error graph
error_fig.update_layout(
    title="Average Error Visualization",
    xaxis_title="Date",
    yaxis_title="Average Error",
    template="plotly_white",
    legend_title="Legend",
    height=600
)
st.plotly_chart(error_fig, use_container_width=True)


# Error Analysis Section
st.header("Error Analysis")
st.write(f"Mean Absolute Error (MAE): **{filtered_df[f'{data_view}_Error'].mean():.2f}**")
st.write(f"Maximum Error: **{filtered_df[f'{data_view}_Error'].max():.2f}**")
st.write(f"Minimum Error: **{filtered_df[f'{data_view}_Error'].min():.2f}**")

# Show Data Table
st.header("Data Table")
st.dataframe(filtered_df)
st.text("")
st.text("")
st.text("")
st.text("")












# Load other data
csv_file2 = "results/pred_vs_actual_sentiment.csv"
df2 = pd.read_csv(csv_file2)

# Convert pubDate to datetime
df2['pubDate'] = pd.to_datetime(df2['pubDate'])

# Extract year, month, and day (ignoring time)
df2['date_only'] = df2['pubDate'].dt.date

# Sort the dataframe by pubDate in ascending order
df2 = df2.sort_values(by='pubDate').reset_index(drop=True)

st.sidebar.text("")
st.sidebar.text("")
st.sidebar.subheader("Model 2 Filter")

# Two separate sliders for start and end dates
start_date = st.sidebar.date_input(
    "Select Start Date:",
    min_value=df2['date_only'].min(),
    max_value=df2['date_only'].max(),
    value=df2['date_only'].min()
)

end_date = st.sidebar.date_input(
    "Select End Date:",
    min_value=df2['date_only'].min(),
    max_value=df2['date_only'].max(),
    value=df2['date_only'].max()
)

data_view2 = st.sidebar.selectbox("Select Feature to View: ", ["SP500_Close", "Two_Days_Later_Close"])

# Filter data based on the selected start and end dates
filtered_df2 = df2[(df2['date_only'] >= start_date) & (df2['date_only'] <= end_date)]

# Aggregate and average only the predicted columns (SP500_Close_Predicted and Two_Days_Later_Close_Predicted) based on date_only
agg_df2 = filtered_df2.groupby('date_only').agg({
    'SP500_Close_Predicted': 'mean',
    'Two_Days_Later_Close_Predicted': 'mean'
}).reset_index()

# Dynamic Graphs: Predicted vs Actual
st.text("")
st.text("")
st.text("")
st.text("")
st.header(f"{(data_view2)}: Predicted vs Actual with Sentiment (Model 2)")
st.text("")
st.markdown(f"The second model utilizes the same input features as the first in addition to a sentiment value derived from this **[Hugging Face model](https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis)** to predict {data_view}.")
fig2 = go.Figure()

# Add predicted and actual values
fig2.add_trace(go.Scatter(
    x=agg_df2['date_only'],
    y=agg_df2[f"{data_view2}_Predicted"],
    mode='lines',
    name='Predicted',
    line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Predicted: %{y}<extra></extra>'
))
fig2.add_trace(go.Scatter(
    x=filtered_df2['date_only'],
    y=filtered_df2[f"{data_view2}_Actual"],
    mode='lines',
    name='Actual',
    line=dict(color='green'),
    hovertemplate='Date: %{x}<br>Actual: %{y}<extra></extra>'
))

# Update layout for Predicted vs Actual graph
fig2.update_layout(
    title=f"{data_view2}: Predicted vs Actual",
    xaxis_title="Date",
    yaxis_title="Value",
    template="plotly_white",
    legend_title="Legend",
    height=600
)

st.plotly_chart(fig2, use_container_width=True)

# Error Graph
st.header("Error Analysis")
error_fig2 = go.Figure()

# Calculate the average error for each date
avg_error_df2 = filtered_df2.groupby('date_only')[f"{data_view2}_Error"].mean().reset_index()

# Add average error values as a bar chart
error_fig2.add_trace(go.Bar(
    x=avg_error_df2['date_only'],
    y=avg_error_df2[f"{data_view2}_Error"],
    name='Average Error',
    marker_color='red',
    hovertemplate='Date: %{x}<br>Average Error: %{y}<extra></extra>'
))

# Update layout for Error graph
error_fig2.update_layout(
    title="Average Error Visualization",
    xaxis_title="Date",
    yaxis_title="Average Error",
    template="plotly_white",
    legend_title="Legend",
    height=600
)
st.plotly_chart(error_fig2, use_container_width=True)


# Error Analysis Section
st.header("Error Analysis")
st.write(f"Mean Absolute Error (MAE): **{filtered_df2[f'{data_view2}_Error'].mean():.2f}**")
st.write(f"Maximum Error: **{filtered_df2[f'{data_view2}_Error'].max():.2f}**")
st.write(f"Minimum Error: **{filtered_df2[f'{data_view2}_Error'].min():.2f}**")

# Show Data Table
st.header("Data Table")
st.dataframe(filtered_df2)
