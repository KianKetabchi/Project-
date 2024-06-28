import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import json
import os
from urllib.error import HTTPError

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

# Page title
st.set_page_config(page_title='Transportation Analysis', page_icon='📊', layout='wide')
st.title('📊 Transportation Analysis')

# About the analysis
with st.expander('About this Analysis'):
    st.markdown("""
    **Purpose of Analysis:**
    1. What/When/How much inventory (Merch & Non-Merch) needs to be transported from Boyne to Progress
    2. How we should build and consolidate pallets.
    
    **Operational Assumptions:**
    - 8 sort points required for Restock (Min-Max + Top-off + ER), Flowthrough, SPO, RVL, Non-merch, and Reserve at Progress.
    - Min-Max & Top-off: 90% of Restock process, can be completed during any shift.
    - ER: the remaining 10% of the restock process, depends on dropped orders.
    - Open Bin / Flowthrough: Should start 1 or 2 hours after receiving.
    - RVL / Reserve / Non-Merch: Slotted on low-volume trips for load smoothing.
    
    **Pallet Build Assumptions:**
    - Using actual case cube of the 52 selected days (4 days in each month).
    - Pallet max Height = 6’.
    - 66% pallet cube utilization considered for Min-Max & Top-off.
    - Use box #82 as Tote for Flowthrough, SPO, Open Bin.
    
    **Truck Assumptions:**
    - Considering box trucks with a capacity of 12 pallets per trip.
    - Multiple box trucks, smaller trucks, or 53’ trailers for transportation.
    """)


file_url = 'https://github.com/KianKetabchi/Project-/raw/main/48%20Days%20of%20FY24.xlsx'

try:
    # Attempt to read Excel file
    all_sheets = pd.read_excel(file_url, sheet_name=None)
    
    # Check if data was successfully loaded
    if all_sheets:
        # Process your data here
        date_options = list(all_sheets.keys())
        selected_date = st.selectbox('Select a Date', date_options)
        
        if selected_date:
            st.write(f"Selected Date: {selected_date}")
    else:
        st.error("Failed to load data from Excel file.")
        
except HTTPError as e:
    st.error(f"HTTPError: {e}")
except ImportError as e:
    st.error(f"ImportError: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

# Sidebar for selecting the date
st.sidebar.subheader('Select Date')
date_options = list(all_sheets.keys())
selected_date = st.sidebar.selectbox('Date', date_options, index=date_options.index(user_settings.get('selected_date', date_options[0])))

# Save selected date
user_settings['selected_date'] = selected_date
save_settings(user_settings)

# Load the data for the selected date
df = all_sheets[selected_date].copy()
df = df.drop("Merch Volume (Cases)", axis=1)

# Set "Truck leaving Boyne" as the index
df.set_index("Time", inplace=True)

# Apply np.ceil() function to specified columns
columns_to_ceil = ['Min-Max/Top-off (Cases)', 'ER (Cases)', 'Open Bin (Cases)',
                   'SPO (Cases)', 'Flowthrough (Cases)', 'RVL (Cases)', 'Reserve (Cases)', 'Non-Merch (Cases)']
df[columns_to_ceil] = np.ceil(df[columns_to_ceil])

# Sidebar for hour range selection
st.sidebar.subheader('Hour Range Selection')

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

# Add an input box for midnight shift percentage within 'Min-Max/Top-off Cases Hour Ranges' section
st.sidebar.markdown("#### Midnight Shift Percentage")
midnight_percentage = st.sidebar.number_input(
    'Enter Percentage for Midnight Shift',
    min_value=0,
    max_value=100,
    value=user_settings.get('midnight_percentage', 80),
    step=5,  # Set the step increment to 5%
    format="%d",
    key='midnight_percentage'
)

# Save the user input for midnight percentage
user_settings['midnight_percentage'] = midnight_percentage
save_settings(user_settings)

# Rest of the hour ranges
st.sidebar.markdown("---")  # Add horizontal line between sliders

hour_ranges_er_cases = get_hour_ranges('ER Cases Hour Ranges', 'er_cases')
st.sidebar.markdown("---")  # Add horizontal line between sliders

hour_ranges_open_bin = get_hour_ranges('Open Bin Cases Hour Ranges', 'open_bin')
st.sidebar.markdown("---")  # Add horizontal line between sliders

hour_ranges_spo = get_hour_ranges('SPO Cases Hour Ranges', 'spo')
st.sidebar.markdown("---")  # Add horizontal line between sliders

hour_ranges_flowthrough = get_hour_ranges('Flowthrough Cases Hour Ranges', 'flowthrough')
st.sidebar.markdown("---")  # Add horizontal line between sliders

hour_ranges_rvl = get_hour_ranges('RVL Cases Hour Ranges', 'rvl')
st.sidebar.markdown("---")  # Add horizontal line between sliders

hour_ranges_reserve = get_hour_ranges('Reserve Cases Hour Ranges', 'reserve')
st.sidebar.markdown("---")  # Add horizontal line between sliders

hour_ranges_non_merch = get_hour_ranges('Non-Merch Cases Hour Ranges', 'non_merch')
st.sidebar.markdown("---")  # Add horizontal line between sliders

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

# Function to adjust other columns based on hour ranges
def adjust_column(column, hour_ranges):
    total_value = df[column].sum()
    total_hours_selected = sum((end - start if start < end else (24 - start + end)) for start, end in hour_ranges)
    adjustment_factor = total_value / total_hours_selected if total_hours_selected > 0 else 0

    for index, row in df.iterrows():
        hour = index.hour
        if is_in_ranges(hour, hour_ranges):
            df.at[index, column] = np.ceil(adjustment_factor)
        else:
            df.at[index, column] = 0


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

# Helper function to check if an hour falls within any of the selected periods
def is_in_ranges(hour, ranges):
    return any(start <= hour < end if start < end else start <= hour or hour < end for start, end in ranges)

            
# Apply adjustments to 'ER (Cases)' column
adjust_er_cases_column(df, hour_ranges_er_cases)


# Apply adjustments to other columns
for col, hour_ranges in [('Open Bin (Cases)', hour_ranges_open_bin),
                         ('SPO (Cases)', hour_ranges_spo),
                         ('Flowthrough (Cases)', hour_ranges_flowthrough),
                         ('RVL (Cases)', hour_ranges_rvl),
                         ('Reserve (Cases)', hour_ranges_reserve),
                         ('Non-Merch (Cases)', hour_ranges_non_merch)]:
    adjust_column(col, hour_ranges)

# Add 'Total (Cases)' column
df['Total (Cases)'] = df[columns_to_ceil].sum(axis=1)

# Convert time to seconds from midnight
def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second

df['Time_in_seconds'] = [time_to_seconds(time) for time in df.index]

# Function to format seconds into time format
def format_seconds_to_time(seconds):
    return str(datetime.timedelta(seconds=seconds))

# Display adjusted table and plot
st.subheader("Analysis Results")

# Create columns layout
col1, col2 = st.columns([4, 3])  # Adjust ratio based on your preference

# Display adjusted table in the first column without 'Time_in_seconds'
with col1:
    st.write("Adjusted Data based on Selected Hour Ranges:")
    st.dataframe(df.drop(columns=['Time_in_seconds']), height=800)  # Make the table bigger

# Combined plot using Plotly in the second column
with col2:
    fig = px.line(
        df.reset_index(),  # Reset index to use 'Time' as x-axis
        x='Time',  # Use 'Time' as x-axis
        y=columns_to_ceil + ['Total (Cases)'],
        title="Combined Plot over Time",
        labels={'value': 'Number of Cases', 'variable': 'Columns', 'Time': 'Time'},
    )

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=df.index,
            ticktext=[format_seconds_to_time(s) for s in df['Time_in_seconds']],
            tickangle=45,
        ),
        autosize=False,
        width=1000,  #
        height=700,  # Adjust plot height as needed
    )

    st.plotly_chart(fig, use_container_width=True)
