import pandas as pd
import numpy as np
import streamlit as st
import datetime
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# File to store settings
settings_file = 'user_settings.json'

def save_settings(settings):
    with open(settings_file, 'w') as f:
        json.dump(settings, f)

def load_settings():
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            return json.load(f)
    return {}

# Load previous settings if available
user_settings = load_settings()
midnight_percentage = user_settings.get('midnight_percentage', 40)  # Set default to 40 if not in settings

# Page title
st.set_page_config(page_title='Transportation Analysis', page_icon='📊', layout='wide')
st.title('📊 Transportation Analysis')

# About the analysis
with st.expander('Purpose of Analysis & Assumptions:'):
    st.markdown("""
    **Purpose of Analysis:**
    1. What/When/How much inventory (Merch & Non-Merch) needs to be transported from Boyne to Progress

    2. How we should build and consolidate pallets.

    **Assumptions:**
    - 8 sort points required for Restock (Min-Max + Top-off + ER), Open Bin, Flowthrough, SPO, RVL, Non-merch, and Reserve at Progress.

    - Restock: Current state -> Top-off + ER + Open Bin putaway is 6% of the number of unique SKUs we sold that day based on Order History data. (Of which ER is 49%, Top-off is 13%, and Open Bin is 38%)  

    - Min-Max & Top-off: Assumed to be 90% of the restock process for the future state. It can be completed during any shift. (Currently, it is mostly completed on the midnight shift.)

    - ER: Assumed to be the remaining 10% of the restock process for the future state. It is dependent on order drop time.

    - Open Bin / Flowthrough: Based on FY24 putaway to pickface and Flowthrough data / Should start 1 or 2 hours after receiving 

    - RVL / Reserve: Based on FY24, E-return data and Merch-pull data for RVL / Reserve should be calculated based on Inventory turns but for now we consider 1 pallet per day (20 cases)

    - Non-Merch: The number of cases we need to send each week divided by 7 (send out the daily volume for non-merch each day)
    """)

with st.expander('Model Features & Functionality:'):
    st.markdown("""
    - The "Select Date" dropdown would specify the day being considered for both the table and plot.          

    - Each order type has a unique slider that shows all 24 hours of each day, which is adjustable for different scenarios.

    - In case of needing to choose a non-continuous time period for any order type, you can check the checkbox for each slider to set a second time period.

    - For Min-Max/Top-off, there is an option to determine what percentage of the total Min-Max/Top-off needs to be done at the midnight shift. It would assign the remaining amount to morning and evening shifts.

    - For ERs, since we aren't notified of order drops in advance, if we choose not to send emergency reshipments (ERs) for a few hours, all the cases that accumulate during those hours will be shipped out with the first subsequent ER shipment. 
    """)

# Load all sheets into a dictionary
file_path = 'C:/Users/kketabchikhonsari/Downloads/Transportation Analysis/48 Days of FY24.xlsx'
all_sheets = pd.read_excel(file_path, sheet_name=None)  # Read all sheets into a dictionary

# Sidebar selection for fiscal year
st.sidebar.subheader('Select Fiscal Year')
year_options = ['FY24', 'FY25', 'FY26', 'FY27', 'FY28', 'FY29', 'FY30']
selected_year = st.sidebar.selectbox('Fiscal Year', year_options)

# Sidebar selection for forecast type
st.sidebar.subheader('Select Forecast Type')
forecast_options = ['Likely Forecast', 'Aggressive Forecast']
selected_forecast = st.sidebar.selectbox('Forecast Type', forecast_options)

# Sidebar for selecting the date
st.sidebar.subheader('Select Date')
date_options = list(all_sheets.keys())
selected_date = st.sidebar.selectbox('Date', date_options, index=date_options.index(user_settings.get('selected_date', date_options[0])))

# Load the data for the selected date
df = all_sheets[selected_date].copy()
df = df.drop(["Merch Volume (Cases)","Trip"], axis=1)

# Convert 'Time' to datetime objects for operations
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

# Set "Truck leaving Boyne" as the index
df.set_index("Time", inplace=True)

# Sidebar checkboxes for column visibility
st.sidebar.subheader('Select Columns to Display')
columns_to_ceil = ['Min-Max/Top-off (Cases)', 'ER (Cases)', 'Open Bin (Cases)',
                   'SPO (Cases)', 'Flowthrough (Cases)', 'RVL (Cases)', 'Reserve (Cases)', 'Non-Merch (Pallets)']
column_visibility = {}
for col in columns_to_ceil:
    column_visibility[col] = st.sidebar.checkbox(col, value=True)

# Calculate and display the sum of 'Total (Cases)' in the sidebar based on selected columns
def calculate_and_round_total_cases_with_new_grid(df, selected_columns):
    # Calculate the sum of the 'Inside of Grid' columns
    df['Total of Inside of Grid'] = df[['Min-Max/Top-off (Cases)', 'ER (Cases)', 'Open Bin (Cases)', 'RVL (Cases)', 'Reserve (Cases)']].sum(axis=1)
    
    # Calculate the sum of the 'Outside of Grid' columns
    df['Total of Outside of Grid'] = df[['SPO (Cases)', 'Flowthrough (Cases)']].sum(axis=1)
    
    # Add 'Non-Merch (Pallets)' column without changes
    df['Non-Merch (Pallets)'] = df['Non-Merch (Pallets)']  # This column reflects actual values from the data
    
    # Sum all cases for each row without rounding
    df['Total (Cases)'] = df[selected_columns].sum(axis=1)
    
    # Calculate the total sum of the 'Total (Cases)' column and round up
    total_cases_sum = np.ceil(df['Total (Cases)'].sum())
    
    return df, total_cases_sum

# Update the total cases based on selected columns
selected_columns = [col for col in columns_to_ceil if column_visibility[col]]
df_filtered, total_cases_sum = calculate_and_round_total_cases_with_new_grid(df, selected_columns)

st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Add blank space
st.sidebar.markdown(f"<span style='font-size: 20px;'><strong>Total Number of Cases:</strong> {int(total_cases_sum):,}</span>", unsafe_allow_html=True)

# Add "Case/Pallet" section based on sheet name
case_pallet_mapping = {
    'March 1': 21, 'March 8': 21, 'March 15': 21, 'March 27': 21,
    'April 4': 22, 'April 12': 22, 'April 18': 22, 'April 26': 22,
    'May 3': 24, 'May 10': 24, 'May 16': 24, 'May 24': 24,
    'June 5': 22, 'June 12': 22, 'June 15': 22, 'June 22': 22,
    'July 5': 23, 'July 10': 23, 'July 19': 23, 'July 31': 23,
    'August 1': 21, 'August 8': 21, 'August 21': 21, 'August 23': 21,
    'September 5': 18, 'September 11': 18, 'September 19': 18, 'September 27': 18,
    'October 5': 18, 'October 10': 18, 'October 17': 18, 'October 30': 18,
    'November 2': 15, 'November 14': 15, 'November 20': 15, 'November 24': 15,
    'December 5': 16, 'December 8': 16, 'December 19': 16, 'December 26': 16,
    'January 7': 16, 'January 10': 16, 'January 16': 16, 'January 25': 16,
    'February 7': 18, 'February 13': 18, 'February 15': 18, 'February 29': 18
}

# Get the case/pallet value based on selected date
case_pallet_value = case_pallet_mapping.get(selected_date, "Unknown")

st.sidebar.markdown(f"<span style='font-size: 20px;'><strong>Case/Pallet:</strong> {case_pallet_value}</span>", unsafe_allow_html=True)

# Add a blank space and title before hour ranges
st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Blank space
st.sidebar.markdown("## Hour Range Selection")

# Input to set midnight percentage
midnight_percentage = st.sidebar.number_input('Midnight Shift Percentage', min_value=0, max_value=100, value=midnight_percentage)

# Save the new midnight percentage value
user_settings['midnight_percentage'] = midnight_percentage
save_settings(user_settings)

# Function to get slider hour range input with an optional second period
def get_hour_ranges(label, key_prefix):
    st.sidebar.markdown(f"{label}")
    ranges = []

    # First time range
    start_hour_1, end_hour_1 = st.sidebar.slider(
        'Select First Period',
        min_value=0,
        max_value=23,
        value=(user_settings.get(f'{key_prefix}_start1', 0), user_settings.get(f'{key_prefix}_end1', 23)),
        format="%d:00",  # Use simple integer format
        key=f"{key_prefix}_first"
    )
    ranges.append((start_hour_1, end_hour_1 + 1))

    # Save settings for the first period
    user_settings[f'{key_prefix}_start1'] = start_hour_1
    user_settings[f'{key_prefix}_end1'] = end_hour_1

    # Checkbox for second time period
    add_second_period = st.sidebar.checkbox(f'Add Second Period for {label}', key=f"{key_prefix}_checkbox", value=user_settings.get(f'{key_prefix}_add_second', False))

    if add_second_period:
        start_hour_2, end_hour_2 = st.sidebar.slider(
            'Select Second Period',
            min_value=0,
            max_value=23,
            value=(user_settings.get(f'{key_prefix}_start2', 0), user_settings.get(f'{key_prefix}_end2', 23)),
            format="%d:00",  # Use simple integer format
            key=f"{key_prefix}_second"
        )
        ranges.append((start_hour_2, end_hour_2 + 1))

        # Save settings for the second period
        user_settings[f'{key_prefix}_start2'] = start_hour_2
        user_settings[f'{key_prefix}_end2'] = end_hour_2

    # Save whether the second period is added or not
    user_settings[f'{key_prefix}_add_second'] = add_second_period

    save_settings(user_settings)
    return ranges

# Assigning unique keys for each time range slider group
hour_ranges_minmax_topoff = get_hour_ranges('Min-Max/Top-off Cases Hour Ranges', 'minmax_topoff')
hour_ranges_er = get_hour_ranges('ER Cases Hour Ranges', 'er')
hour_ranges_open_bin = get_hour_ranges('Open Bin Cases Hour Ranges', 'open_bin')
hour_ranges_spo = get_hour_ranges('SPO Cases Hour Ranges', 'spo')
hour_ranges_flowthrough = get_hour_ranges('Flowthrough Cases Hour Ranges', 'flowthrough')
hour_ranges_rvl = get_hour_ranges('RVL Cases Hour Ranges', 'rvl')
hour_ranges_reserve = get_hour_ranges('Reserve Cases Hour Ranges', 'reserve')
# The slider for 'Non-Merch (Pallets)' has been removed as it should reflect actual values only

# Function to check if an hour falls within any of the selected periods
def is_in_ranges(hour, ranges):
    return any(start <= hour < end if start < end else start <= hour or hour < end for start, end in ranges)

# Function to adjust 'Min-Max/Top-off (Cases)' column based on midnight percentage and hour ranges
def adjust_minmax_topoff_column(df, midnight_percentage, hour_ranges):
    # Calculate total sum of 'Min-Max/Top-off (Cases)'
    total_sum = df['Min-Max/Top-off (Cases)'].sum()

    # Calculate the amount for midnight shift and remaining
    midnight_amount = np.ceil((midnight_percentage / 100) * total_sum)
    remaining_amount = np.ceil(total_sum - midnight_amount)

    # Define midnight and day hours
    midnight_hours = list(range(22, 24)) + list(range(0, 6))
    midnight_range = [hr for hr in midnight_hours if is_in_ranges(hr, hour_ranges)]

    day_hours = [hr for hr in range(24) if hr not in midnight_hours]
    day_range = [hr for hr in day_hours if is_in_ranges(hr, hour_ranges)]

    # Calculate the number of hours in each range
    total_midnight_hours = len(midnight_range)
    total_day_hours = len(day_range)

    # Distribute values
    if total_midnight_hours > 0:
        midnight_value_per_hour = np.ceil(midnight_amount / total_midnight_hours)
    else:
        midnight_value_per_hour = 0

    if total_day_hours > 0:
        day_value_per_hour = np.ceil(remaining_amount / total_day_hours)
    else:
        day_value_per_hour = 0

    # Adjust the column values
    for index, row in df.iterrows():
        hour = index.hour
        if hour in midnight_range:
            df.at[index, 'Min-Max/Top-off (Cases)'] = midnight_value_per_hour
        elif hour in day_range:
            df.at[index, 'Min-Max/Top-off (Cases)'] = day_value_per_hour
        else:
            df.at[index, 'Min-Max/Top-off (Cases)'] = 0

# Apply the adjustment function to 'Min-Max/Top-off (Cases)'
adjust_minmax_topoff_column(df, midnight_percentage, hour_ranges_minmax_topoff)

# Function to split values evenly based on hour ranges
def split_values_evenly(df, column_name, hour_ranges):
    original_values = df[column_name].copy()  # Get original values
    total_value = original_values.sum()  # Get total value
    selected_hours = set(hour for start, end in hour_ranges for hour in range(start, end))

    # Convert index to DatetimeIndex if it isn't already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, format='%H:%M:%S')

    if selected_hours:
        value_per_hour = np.ceil(total_value / len(selected_hours))
        for hour in range(24):  # Iterate through all hours in a day
            if hour in selected_hours:
                df.loc[df.index.hour == hour, column_name] = value_per_hour
            else:
                df.loc[df.index.hour == hour, column_name] = 0

    return df[column_name]

# Apply the split values function to the other columns (except 'Non-Merch (Pallets)')
df['Open Bin (Cases)'] = split_values_evenly(df, 'Open Bin (Cases)', hour_ranges_open_bin)
df['SPO (Cases)'] = split_values_evenly(df, 'SPO (Cases)', hour_ranges_spo)
df['Flowthrough (Cases)'] = split_values_evenly(df, 'Flowthrough (Cases)', hour_ranges_flowthrough)
df['RVL (Cases)'] = split_values_evenly(df, 'RVL (Cases)', hour_ranges_rvl)
df['Reserve (Cases)'] = split_values_evenly(df, 'Reserve (Cases)', hour_ranges_reserve)
# No change applied to 'Non-Merch (Pallets)' as it reflects actual values

# Function to adjust 'ER (Cases)' column based on selected hours
def adjust_er_cases_column(df, hour_ranges):
    unselected_sum = 0
    added_to_next_hour = False

    for index, row in df.iterrows():
        hour = index.hour

        if is_in_ranges(hour, hour_ranges):
            if unselected_sum > 0 and not added_to_next_hour:
                # Find the hour and add unselected_sum to it
                try:
                    df.at[index, 'ER (Cases)'] += unselected_sum
                except KeyError:
                    pass  # Handle if index is not found
                unselected_sum = 0  # Reset the accumulated sum
                added_to_next_hour = True

            # Keep the current value for selected hours
            try:
                df.at[index, 'ER (Cases)'] = df.at[index, 'ER (Cases)']
            except KeyError:
                pass  # Handle if index is not found

            added_to_next_hour = False
        else:
            # Accumulate the sum for unselected hours
            unselected_sum += df.at[index, 'ER (Cases)']
            # Set the current hour value to zero for unselected hours
            try:
                df.at[index, 'ER (Cases)'] = 0
            except KeyError:
                pass  # Handle if index is not found

    # Edge case: If unselected_sum remains at the end, add it to the earliest hour available
    if unselected_sum > 0 and not added_to_next_hour:
        first_hour_index = 0
        while first_hour_index < len(df) and df.iloc[first_hour_index]['ER (Cases)'] == 0:
            first_hour_index += 1

        if first_hour_index < len(df):
            try:
                df.at[df.index[first_hour_index], 'ER (Cases)'] += unselected_sum
            except KeyError:
                pass  # Handle if index is not found

# Round up 'ER (Cases)' column before applying the logic
df['ER (Cases)'] = df['ER (Cases)'].apply(np.ceil)

# Apply adjustments to 'ER (Cases)' column
adjust_er_cases_column(df, hour_ranges_er)

# Define coefficients for each column and fiscal year for Likely Forecast
likely_forecast_coefficients = {
    'Min-Max/Top-off (Cases)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.04, 'FY27': 1.19, 'FY28': 1.3, 'FY29': 1.38, 'FY30': 1.49},
    'ER (Cases)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.04, 'FY27': 1.19, 'FY28': 1.3, 'FY29': 1.38, 'FY30': 1.49},
    'Open Bin (Cases)': {'FY24': 1, 'FY25': 1.07, 'FY26': 1.34, 'FY27': 1.54, 'FY28': 1.67, 'FY29': 1.77, 'FY30': 1.92},
    'SPO (Cases)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.04, 'FY27': 1.19, 'FY28': 1.3, 'FY29': 1.38, 'FY30': 1.49},
    'Flowthrough (Cases)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.04, 'FY27': 1.19, 'FY28': 1.3, 'FY29': 1.38, 'FY30': 1.49},
    'RVL (Cases)': {'FY24': 1, 'FY25': 0.62, 'FY26': 0.66, 'FY27': 0.68, 'FY28': 0.7, 'FY29': 0.7, 'FY30': 0.71},
    'Reserve (Cases)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.04, 'FY27': 1.19, 'FY28': 1.3, 'FY29': 1.38, 'FY30': 1.49},
    'Non-Merch (Pallets)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.04, 'FY27': 1.19, 'FY28': 1.3, 'FY29': 1.38, 'FY30': 1.49},
}

# Define coefficients for each column and fiscal year for Aggressive Forecast
aggressive_forecast_coefficients = {
    'Min-Max/Top-off (Cases)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.22, 'FY27': 1.43, 'FY28': 1.70, 'FY29': 2.03, 'FY30': 2.49},
    'ER (Cases)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.22, 'FY27': 1.43, 'FY28': 1.70, 'FY29': 2.03, 'FY30': 2.49},
    'Open Bin (Cases)': {'FY24': 1, 'FY25': 1.07, 'FY26': 1.59, 'FY27': 1.86, 'FY28': 2.21, 'FY29': 2.65, 'FY30': 3.24},
    'SPO (Cases)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.22, 'FY27': 1.43, 'FY28': 1.70, 'FY29': 2.03, 'FY30': 2.49},
    'Flowthrough (Cases)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.22, 'FY27': 1.43, 'FY28': 1.70, 'FY29': 2.03, 'FY30': 2.49},
    'RVL (Cases)': {'FY24': 1, 'FY25': 0.62, 'FY26': 0.78, 'FY27': 0.82, 'FY28': 0.88, 'FY29': 0.92, 'FY30': 0.97},
    'Reserve (Cases)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.22, 'FY27': 1.43, 'FY28': 1.70, 'FY29': 2.03, 'FY30': 2.49},
    'Non-Merch (Pallets)': {'FY24': 1, 'FY25': 0.79, 'FY26': 1.22, 'FY27': 1.43, 'FY28': 1.70, 'FY29': 2.03, 'FY30': 2.49},
}

# Determine which coefficients to use based on selected forecast type
if selected_forecast == 'Likely Forecast':
    coefficients = likely_forecast_coefficients
else:
    coefficients = aggressive_forecast_coefficients

# Function to project data for a given fiscal year
def project_data_for_year(df, year, coefficients):
    projected_df = df.copy()
    for column in projected_df.columns:
        if column in coefficients:
            coefficient = coefficients[column].get(year, 1.0)  # Default coefficient is 1.0 if not specified
            projected_df[column] *= coefficient
    return projected_df

# Calculate projected data for the selected year
projected_df = project_data_for_year(df, selected_year, coefficients)

# Recalculate 'Total (Cases)', 'Total of Inside of Grid', and 'Total of Outside of Grid' dynamically based on selected columns and round up total
df_filtered, total_cases_sum = calculate_and_round_total_cases_with_new_grid(projected_df, selected_columns)

# Create the filtered DataFrame based on column visibility and include the new columns
df_filtered_display = df_filtered[selected_columns + ['Total of Inside of Grid', 'Total of Outside of Grid', 'Total (Cases)']]

# Keep the exact values but display as integers in the table
def display_value(value):
    if 0 < value < 1:
        return 1  # Display as 1 if between 0 and 1
    return int(value)  # Display the integer value otherwise

df_filtered_display = df_filtered_display.applymap(display_value)

# Reset index to use 'Time' as a column with the correct format
df_filtered_display = df_filtered_display.reset_index()
df_filtered_display['Time'] = df_filtered_display['Time'].apply(lambda x: x.strftime('%H:%M:%S'))

# Center align the table content and headers
df_filtered_style = df_filtered_display.style.set_properties(**{'text-align': 'center'})
df_filtered_style = df_filtered_style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

# Display the filtered table with scrollable view and fixed header
st.subheader(f'Summary - Based on Number of Cases for {selected_year} ({selected_forecast})')
st.markdown(
    """
    <style>
    .dataframe-container-first {
        max-height: 250px; /* Adjust height to show fewer rows */
        overflow-y: auto;
        position: relative;
    }
    .dataframe-container-first table {
        width: 100%;
        border-collapse: collapse;
    }
    .dataframe-container-first th, .dataframe-container-first td {
        text-align: center;
        padding: 8px;
        border: 1px solid #ddd;
    }
    .dataframe-container-first th {
        position: sticky;
        top: 0;
        background-color: white;
        z-index: 1;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<div class="dataframe-container-first">{}</div>'.format(df_filtered_style.to_html(index=False)), unsafe_allow_html=True)

# Add download button for the filtered table
# Convert df_filtered_display to CSV, including the Time column
csv = df_filtered_display.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name=f'filtered_data_{selected_year}_{selected_forecast}.csv',
    mime='text/csv',
)

# Define colors for each column
colors = {
    'Min-Max/Top-off (Cases)': '#1f77b4',  # Blue
    'ER (Cases)': '#ff7f0e',  # Orange
    'Open Bin (Cases)': '#2ca02c',  # Green
    'SPO (Cases)': '#d62728',  # Red
    'Flowthrough (Cases)': '#9467bd',  # Purple
    'RVL (Cases)': '#8c564b',  # Brown
    'Reserve (Cases)': '#e377c2',  # Pink
    'Non-Merch (Pallets)': '#7f7f7f',  # Gray
    'Total (Cases)': '#17becf'  # Teal
}

# Define line styles for each column
line_styles = {
    'Total (Cases)': {'width': 3, 'dash': 'solid'}
}

# Plot with fixed colors and styles
fig = go.Figure()
for column in selected_columns + ['Total of Inside of Grid', 'Total of Outside of Grid', 'Total (Cases)']:
    fig.add_trace(go.Scatter(
        x=df_filtered_display['Time'],
        y=df_filtered[column],
        mode='lines',
        name=column,
        line=dict(color=colors.get(column, '#000000'), **line_styles.get(column, {}))
    ))

fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Cases',
    height=600,  # Adjust height as needed (multiplied by 1.5)
    width=2000,  # Adjust width as needed
    margin=dict(r=10)  # Add some space at the end of the plot
)

st.plotly_chart(fig)

# Define fiscal year and forecast type options
year_options = ['FY24', 'FY25', 'FY26', 'FY27', 'FY28', 'FY29', 'FY30']
forecast_options = ['Likely Forecast', 'Aggressive Forecast']

# Initialize the new table DataFrame for Inside of Grid
inside_grid_table_df = pd.DataFrame()

# Initialize the new table DataFrame for Outside of Grid
outside_grid_table_df = pd.DataFrame()

# Initialize the new table DataFrame for Non-Merch (Pallets)
non_merch_pallets_df = pd.DataFrame()

# Loop through each date in the excel file
for date in date_options:
    # Load the data for the date
    date_df = all_sheets[date].copy()
    date_df = date_df.drop(["Merch Volume (Cases)","Trip"], axis=1)
    date_df['Time'] = pd.to_datetime(date_df['Time'], format='%H:%M:%S').dt.time
    date_df.set_index("Time", inplace=True)

    # Convert to DatetimeIndex if not already
    if not isinstance(date_df.index, pd.DatetimeIndex):
        date_df.index = pd.to_datetime(date_df.index, format='%H:%M:%S')

    # Apply the same transformations as the first table
    adjust_minmax_topoff_column(date_df, midnight_percentage, hour_ranges_minmax_topoff)
    date_df['Open Bin (Cases)'] = split_values_evenly(date_df, 'Open Bin (Cases)', hour_ranges_open_bin)
    date_df['SPO (Cases)'] = split_values_evenly(date_df, 'SPO (Cases)', hour_ranges_spo)
    date_df['Flowthrough (Cases)'] = split_values_evenly(date_df, 'Flowthrough (Cases)', hour_ranges_flowthrough)
    date_df['RVL (Cases)'] = split_values_evenly(date_df, 'RVL (Cases)', hour_ranges_rvl)
    date_df['Reserve (Cases)'] = split_values_evenly(date_df, 'Reserve (Cases)', hour_ranges_reserve)
    # 'Non-Merch (Pallets)' is not adjusted as it should reflect actual values
    date_df['ER (Cases)'] = date_df['ER (Cases)'].apply(np.ceil)
    adjust_er_cases_column(date_df, hour_ranges_er)

    # Project data for the selected fiscal year and forecast type
    projected_date_df = project_data_for_year(date_df, selected_year, coefficients)

    # Calculate the total cases for the projected data
    projected_date_df, total_cases_sum = calculate_and_round_total_cases_with_new_grid(projected_date_df, selected_columns)

    # Extract the 'Total of Inside of Grid' and 'Total of Outside of Grid' columns
    inside_grid_cases = np.ceil(projected_date_df['Total of Inside of Grid']).astype(int)
    outside_grid_cases = np.ceil(projected_date_df['Total of Outside of Grid']).astype(int)
    non_merch_pallets = np.ceil(projected_date_df['Non-Merch (Pallets)']).astype(int)  # Extract 'Non-Merch (Pallets)'

    # Add the 'Total of Inside of Grid' to the inside_grid_table_df
    if inside_grid_table_df.empty:
        inside_grid_table_df['Time'] = projected_date_df.index.strftime('%H:%M:%S')
    inside_grid_table_df[date] = inside_grid_cases.values

    # Add the 'Total of Outside of Grid' to the outside_grid_table_df
    if outside_grid_table_df.empty:
        outside_grid_table_df['Time'] = projected_date_df.index.strftime('%H:%M:%S')
    outside_grid_table_df[date] = outside_grid_cases.values

    # Add the 'Non-Merch (Pallets)' to the non_merch_pallets_df
    if non_merch_pallets_df.empty:
        non_merch_pallets_df['Time'] = projected_date_df.index.strftime('%H:%M:%S')
    non_merch_pallets_df[date] = non_merch_pallets.values

# Display the Inside of Grid table
st.subheader(f'Total of Inside of Grid Summary for Different Dates in {selected_year} ({selected_forecast})')
st.write(inside_grid_table_df)

# Add download button for the Inside of Grid table
inside_grid_csv = inside_grid_table_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Total of Inside of Grid Summary as CSV",
    data=inside_grid_csv,
    file_name=f'total_inside_grid_summary_{selected_year}_{selected_forecast}.csv',
    mime='text/csv',
)

# Display the Outside of Grid table
st.subheader(f'Total of Outside of Grid Summary for Different Dates in {selected_year} ({selected_forecast})')
st.write(outside_grid_table_df)

# Add download button for the Outside of Grid table
outside_grid_csv = outside_grid_table_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Total of Outside of Grid Summary as CSV",
    data=outside_grid_csv,
    file_name=f'total_outside_grid_summary_{selected_year}_{selected_forecast}.csv',
    mime='text/csv',
)

# Display the Non-Merch (Pallets) table
st.subheader(f'Non-Merch (Pallets) Summary for Different Dates in {selected_year} ({selected_forecast})')
st.write(non_merch_pallets_df)

# Add download button for the Non-Merch (Pallets) table
non_merch_pallets_csv = non_merch_pallets_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Non-Merch (Pallets) Summary as CSV",
    data=non_merch_pallets_csv,
    file_name=f'non_merch_pallets_summary_{selected_year}_{selected_forecast}.csv',
    mime='text/csv',
)
