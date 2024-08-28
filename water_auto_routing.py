import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import altair as alt

st.set_page_config(page_title="Water Optimization", page_icon=💧)
# Set up layout
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout='wide')
# Set up Margin
st.markdown("""
    <style>
    .css-z5fcl4 {
        width: 100%;
        padding: 1rem 2rem 5rem;
        min-width: auto;
        max-width: initial;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
    <style>
    .css-1ywmfj8 {
    width: 1370px;
    position: relative;
    display: flex;
    flex: 1 1 0%;
    flex-direction: column;
    gap: 0rem;
}
    </style>
"""
,unsafe_allow_html=True
) 

# Load data
data_water = pd.read_csv('water_data_update.csv', sep=';')
data_water['date'] = pd.to_datetime(data_water['date'], format="%d/%m/%y")

# Get the latest date for model calculation
latest_date = data_water['date'].max()
latest_data = data_water[data_water['date'] == latest_date]

# Function for penalized success rate
def y2_func(x, decay, scale, origin, offset):
    sigma = scale / np.sqrt(-2 * np.log(decay))
    distance = np.maximum(0, np.abs(x - origin) - offset)
    decay_value = np.exp(- (distance ** 2) / (2 * (sigma ** 2)))
    return x * decay_value

# Function to calculate penalized success rates
def calculate_penalized_SR(data, SKU, success_rates, decay, scale, origin, offset):
    SKU_data = data[data['sku'] == SKU].copy()
    SKU_data.sort_values(by=['goods_id'], inplace=True)
    SKU_data['success_rate'] = success_rates
    penalized_SR = [y2_func(x, decay, scale, origin, offset) for x in success_rates]
    SKU_data['penalized_SR'] = penalized_SR
    return SKU_data

# Main layout
st.header(":droplet: WATER'S AGGREGATOR - OPTIMIZATION SIMULATOR")
st.subheader(":clipboard: Input Parameters and Success Rates ")

# Input for penalization parameters

# st.markdown("<h4 style='text-align: left; font-size: 1.2em;'>Penalized Success Rate Parameters:</h4>", unsafe_allow_html=True)
selected_SKU, quantity = st.columns(2)
# Select SKU
with selected_SKU:
    sku_options = latest_data['sku'].unique()
    default_sku = list(sku_options).index('pdam_palyja') if 'pdam_palyja' in sku_options else 0
    selected_SKU = st.selectbox("**Choose The SKU**", sku_options, index=default_sku)
# Select Params
with quantity:
    quantity = st.number_input("**Estimate Total Number of Transactions**", min_value=1, value=1000, step=1)

# Params for Decay Function
decay = 0.5
scale = 60
origin = 100
offset = 0

original_success_rates = []
SKU_data = latest_data[latest_data['sku'] == selected_SKU].copy()

# Determines the number of aggreagators
num_aggregators = len(SKU_data)
columns = st.columns(num_aggregators)

# Dynamic columns
for i, (index, row) in enumerate(SKU_data.iterrows()):
    index = i % num_aggregators
    with columns[index]:
        rate = st.number_input(f"**Input Success Rate for {row['aggregator']} Aggregator**", value=row['success_rate'], key=f"rate_{i}")
        original_success_rates.append(rate)

# Calculate and display results using user inputs
default_allocations = {
    'indobest': 5,
    'ids': 10,
    'dji': 15,
    'fortuna': 15,
    'syb': 5,
    'bimasakti': 15,
    'mkm': 15,
    'mitracomm': 10
}

# Map default allocations to the current SKU data
SKU_data['default_allocation'] = SKU_data['aggregator'].map(default_allocations)
# Rebase the default allocations to sum to 100% for available aggregators
available_allocations = SKU_data['default_allocation'].dropna()
total_allocation = available_allocations.sum()
rebased_allocations = (available_allocations / total_allocation) * 100
SKU_data['rebased_allocation'] = rebased_allocations

result_df = calculate_penalized_SR(latest_data, selected_SKU, original_success_rates, decay, scale, origin, offset)
# Add default allocation to result_df
result_df = result_df.merge(SKU_data[['aggregator', 'rebased_allocation']], on='aggregator', how='left')

# Optimization with updated success rates
goods_ids = result_df['goods_id'].tolist()
MDR_values = result_df['mdr_amount'].values
penalized_SR_values = result_df['penalized_SR'].values

p_x = cp.Variable(len(original_success_rates), nonneg=True)
objective = cp.Maximize(cp.sum(cp.multiply(MDR_values, cp.multiply(penalized_SR_values, p_x))))
constraints = [p_x >= 3, cp.sum(p_x) == 100]
problem = cp.Problem(objective, constraints)
problem.solve()

result_df['optimum_allocation'] = p_x.value
default_allocation_percent = result_df['rebased_allocation'].values
benchmark_revenue = quantity * np.sum(MDR_values * penalized_SR_values * default_allocation_percent)
optimal_revenue = quantity * np.sum(MDR_values * penalized_SR_values * p_x.value)
different_revenue = optimal_revenue - benchmark_revenue

st.write("---")
# MAIN LAYOUT 
charts, dataframes = st.columns([2, 2])
with charts:
    st.subheader(f":bar_chart: Daily Success Rates for {selected_SKU}")
    SKU_data = latest_data[latest_data['sku'] == selected_SKU].copy()
    # Plotting the bar chart for daily success rates by aggregator
    all_dates_SKU_data = data_water[data_water['sku'] == selected_SKU]
    daily_success_rates = all_dates_SKU_data.pivot_table(index='date', columns='aggregator', values='success_rate').reset_index()
    
   
    # Melt the data for Altair
    melted_data = daily_success_rates.melt(id_vars='date', var_name='aggregator', value_name='success_rate')
    
    bars = alt.Chart(melted_data).mark_bar().encode(
        x=alt.X('yearmonthdate(date):T', axis=alt.Axis(title='Date',labelAlign='center', format="%b %d, %Y")),
        y=alt.Y('success_rate:Q', axis=alt.Axis(title='Success Rate'), scale=alt.Scale(domain=[0, 120])),
        color=alt.Color('aggregator:N', scale=alt.Scale(scheme='set2'), legend=alt.Legend(orient='bottom', title=None)),
        xOffset='aggregator:N',
        tooltip=['date:T', 'aggregator:N', 'success_rate:Q']
    ).properties(
        width=800,
        height=400
    )

    text = bars.mark_text(
        align='center',
        baseline='middle',
        dy=-10  
    ).encode(
        text='success_rate:Q'
    )

    chart = alt.layer(bars, text).properties(
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_legend(
        titleFontSize=14,
        labelFontSize=12
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_legend(
        titleFontSize=14,
        labelFontSize=14
    )
    
    # Display the plot in Streamlit
    st.altair_chart(chart, use_container_width=True)

with dataframes:
    st.subheader(":page_facing_up: Aggregators Details")
    columns_to_display = ['aggregator', 'mdr_amount', 'rebased_allocation', 'optimum_allocation']
    column_aliases = {
        'aggregator': 'Aggregator',
        'mdr_amount': 'MDR Amount (Rp)',
        'rebased_allocation': 'Default Allocation (%)',
        'optimum_allocation': 'Optimized Allocation (%)'
    }

    SKU_data_filtered = result_df[columns_to_display].rename(columns=column_aliases)
    SKU_data_filtered.reset_index(drop=True, inplace=True)

    # Styling function
    def style_dataframe(df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        return df.style\
            .set_table_styles(
                [{'selector': 'th',
                  'props': [('background-color', '#f9f8fa'),
                            ('color', '#333333'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center'),
                            ('border', '1px solid #ddd')]}])\
            .highlight_max(subset=numeric_columns, color='#ffddc1')\
            .highlight_min(subset=numeric_columns, color='#c1e1c5')\
            .format({"MDR Amount (Rp)": "Rp {:,.0f}", 
                     "Default Allocation (%)": "{:.1f}%", 
                     "Optimized Allocation (%)": "{:.1f}%"})

    # Convert styled DataFrame to HTML and set full width
    styled_df_html = style_dataframe(SKU_data_filtered).set_table_attributes('style="width:100%"').to_html()

    # Display the styled DataFrame in Streamlit
    st.markdown(
        f"""
        <div style="overflow-x:auto; width:100%;">
            {styled_df_html}
        """, unsafe_allow_html=True
    )
# with dataframes:
#     st.subheader(":page_facing_up: Aggregators Details")
#     columns_to_display = ['aggregator', 'mdr_amount', 'rebased_allocation', 'optimum_allocation']
#     column_aliases = {
#         'aggregator': 'Aggregator',
#         'mdr_amount': 'MDR Amount (Rp)',
#         'rebased_allocation': 'Default Allocation (%)',
#         'optimum_allocation': 'Optimized Allocation (%)'
#     }

#     SKU_data_filtered = result_df[columns_to_display].rename(columns=column_aliases)
#     SKU_data_filtered.reset_index(drop=True, inplace=True)

#     # Define a styling function
#     def style_dataframe(df):
#         return df.style\
#             .set_table_styles(
#                 [{'selector': 'th',
#                   'props': [('background-color', '#f4f4f4'),
#                             ('color', '#333333'),
#                             ('font-weight', 'bold'),
#                             ('text-align', 'center'),
#                             ('border', '1px solid #ddd')]}])\
#             .highlight_max(color='#ffddc1')\
#             .highlight_min(color='#c1e1c5')\
#             .format({"MDR Amount (Rp)": "Rp {:,.0f}", 
#                      "Default Allocation (%)": "{:.2f}%", 
#                      "Optimized Allocation (%)": "{:.2f}%"})

#     # Display the styled DataFrame in Streamlit
#     st.write(style_dataframe(SKU_data_filtered).to_html(), unsafe_allow_html=True)

# with dataframes:
#     st.subheader(":page_facing_up: Aggregators Details")
#     columns_to_display = ['aggregator', 'mdr_amount', 'rebased_allocation', 'optimum_allocation']
#     column_aliases = {
#         'aggregator': 'Aggregator',
#         'mdr_amount': 'MDR Amount (Rp)',
#         'rebased_allocation': 'Default Allocation (%)',
#         'optimum_allocation': 'Optimized Allocation (%)'
#     }
#     SKU_data_filtered = result_df[columns_to_display].rename(columns=column_aliases)
#     SKU_data_filtered.reset_index(drop=True, inplace=True)
#     st.dataframe(SKU_data_filtered, use_container_width=True)

    ## Revenue
    st.subheader(":chart_with_upwards_trend: Revenue Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div style="background-color:#f9f8fa; padding: 10px; border-radius: 8px; text-align: center; width: auto;">
                <h4>Default Revenue</h4>
                <p style="font-size: 20px; color: #FF7F50; margin: 0;"><b>Rp {:,.0f}</b></p>
                <small>Default Allocation</small>
            </div>
            """.format(benchmark_revenue), unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div style="background-color:#f9f8fa; padding: 10px; border-radius: 8px; text-align: center; width: auto;">
                <h4>Optimized Revenue</h4>
                <p style="font-size: 20px; color: #FF7F50; margin: 0;"><b>Rp {:,.0f}</b></p>
                <small>Optimized Allocation</small>
            </div>
            """.format(optimal_revenue), unsafe_allow_html=True)

    with col3:
        st.markdown(
            """
            <div style="background-color:#f9f8fa; padding: 10px; border-radius: 8px; text-align: center; width: auto;">
                <h4>Difference in Revenue</h4>
                <p style="font-size: 20px; color: #FF7F50; margin: 0;"><b>Rp {:,.0f}</b></p>
                <small>Difference</small>
            </div>
            """.format(different_revenue), unsafe_allow_html=True)


