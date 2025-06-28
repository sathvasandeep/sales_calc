import streamlit as st
import pandas as pd
import numpy as np
import math

# --- Configuration and Page Title ---
st.set_page_config(layout="wide", page_title="SaaS Sales & Revenue Forecaster")
st.title("📈 SaaS Sales & Revenue Forecaster")
st.caption("Adjust inputs in the sidebar to see projections change.")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Business Inputs")

total_revenue_target_lakhs = st.sidebar.number_input(
    "Overall Annual Revenue Target (Lakhs)", min_value=100, value=400, step=50
)
fy_start_month = st.sidebar.selectbox(
    "Financial Year Start Month",
    ["Apr","May","Jun","Jul","Aug","SEP","OCT","Nov","Dec", "Jan","Feb","Mar"], # Add others if needed
    index=0 # Default to April
)

# --- THIS IS THE CORRECTED BLOCK ---
# --- THIS VERSION REMOVES format_func ENTIRELY ---
current_month_index = st.sidebar.selectbox(
    "Select Current Month (of FY)",
    options=list(range(1, 13)),
    # format_func=...  <--- THE WHOLE LINE IS DELETED
    index=0
)
# --- END OF MODIFIED BLOCK ---
# --- END OF CORRECTED BLOCK ---


st.sidebar.header("💰 Revenue & Deal Size")
avg_ticket_size_lakhs = st.sidebar.number_input(
    "Average Ticket Size (Lakhs)", min_value=1.0, value=10.0, step=0.5
)
spillover_lakhs = st.sidebar.number_input(
    "Spillover Revenue from Last FY (Lakhs)", min_value=0.0, value=70.0, step=5.0
)
renewals_lakhs = st.sidebar.number_input(
    "Expected Renewal Revenue This FY (Lakhs)", min_value=0.0, value=30.0, step=5.0
)

st.sidebar.header("⏳ Sales Cycle")
avg_sales_cycle_months = st.sidebar.slider(
    "Average Sales Cycle (Months)", min_value=3, max_value=18, value=9, step=1
)

st.sidebar.header("📊 Lead & Conversion Metrics")
lead_to_deal_conversion_rate_percent = st.sidebar.slider(
    "Overall Lead-to-Deal Conversion Rate (%)", min_value=0.1, max_value=15.0, value=3.2, step=0.1
)
cpl_rupees = st.sidebar.number_input(
    "Cost Per Lead (CPL) (₹)", min_value=100, value=3000, step=100
)
st.sidebar.markdown("---")
leads_already_generated = st.sidebar.number_input(
    "Leads Generated *Before* This FY Start (Estimate)", min_value=0, value=61, step=1
)


# --- Core Calculations ---

# Convert Lakhs/Crores to absolute values
total_revenue_target = total_revenue_target_lakhs * 100000
avg_ticket_size = avg_ticket_size_lakhs * 100000
spillover_revenue = spillover_lakhs * 100000
renewals_revenue = renewals_lakhs * 100000

# Calculate Net New Target for the FY
known_revenue = spillover_revenue + renewals_revenue
net_new_target = max(0, total_revenue_target - known_revenue) # Ensure not negative
net_new_clients_needed_total = math.ceil(net_new_target / avg_ticket_size) if avg_ticket_size > 0 else 0

# Calculate Leads per Deal
if lead_to_deal_conversion_rate_percent > 0:
    conversion_rate_decimal = lead_to_deal_conversion_rate_percent / 100
    leads_per_deal = math.ceil(1 / conversion_rate_decimal)
else:
    conversion_rate_decimal = 0
    leads_per_deal = float('inf') # Avoid division by zero, indicate infinite leads needed

# Calculate Revenue Expected from Already Generated Leads (closing this FY)
# Assuming they follow the average sales cycle from their generation time (roughly Feb/Mar)
revenue_from_prior_leads = leads_already_generated * conversion_rate_decimal * avg_ticket_size
clients_from_prior_leads = math.ceil(leads_already_generated * conversion_rate_decimal)

# Calculate Remaining Gap to be filled by leads generated *this* FY
revenue_gap_for_new_leads = max(0, net_new_target - revenue_from_prior_leads)
clients_needed_from_new_leads = math.ceil(revenue_gap_for_new_leads / avg_ticket_size) if avg_ticket_size > 0 else 0

# Determine the window for generating leads that can close *within* this FY
# Leads generated in month `gen_month` close in `gen_month + cycle`. We need `gen_month + cycle <= 12`.
# So, the last generation month index (1-based) is `12 - cycle`.
last_gen_month_index_for_fy = max(0, 12 - avg_sales_cycle_months) # Month index (1-based) by which leads must be generated


# Calculate how many months are left *in that window*, considering the current month
# If current month is past the last gen month, window is closed (0 months left)
months_left_in_window = max(0, last_gen_month_index_for_fy - current_month_index + 1) if current_month_index <= last_gen_month_index_for_fy else 0


# Calculate Total Leads needed during the remaining window & Monthly Rate
if months_left_in_window > 0 and leads_per_deal != float('inf') and clients_needed_from_new_leads > 0:
    total_leads_needed_in_window = clients_needed_from_new_leads * leads_per_deal
    monthly_leads_needed_in_window = math.ceil(total_leads_needed_in_window / months_left_in_window)
    monthly_spend_in_window = monthly_leads_needed_in_window * cpl_rupees
    needs_met = True
elif clients_needed_from_new_leads <= 0: # Target already met by prior leads/known revenue
    total_leads_needed_in_window = 0
    monthly_leads_needed_in_window = 0
    monthly_spend_in_window = 0
    needs_met = True # Technically met, no new leads required for *this* FY target
else:
    # It's too late, conversion is zero, or clients needed > 0 but window is closed/invalid
    total_leads_needed_in_window = float('inf') if clients_needed_from_new_leads > 0 else 0
    monthly_leads_needed_in_window = float('inf') if clients_needed_from_new_leads > 0 else 0
    monthly_spend_in_window = float('inf') if clients_needed_from_new_leads > 0 else 0
    needs_met = False # Indicate target might not be reachable within FY based on inputs

# --- Display Key Metrics ---
st.header("📊 Key Performance Indicators (KPIs)")
col1, col2, col3 = st.columns(3)
col1.metric("Net New Target (FY)", f"₹{net_new_target/100000:.2f} L", delta=f"{net_new_clients_needed_total} Clients Total")
col2.metric("Leads Needed Per Deal", f"{leads_per_deal:.0f}", help=f"Based on {lead_to_deal_conversion_rate_percent}% conversion")
col3.metric("Revenue / Client", f"₹{avg_ticket_size/100000:.2f} L")

st.markdown("---")
st.header("🎯 Lead Generation Required (To Hit FY Target)")

# Determine FY end month string for messages
fy_end_dt = pd.Timestamp(f'2026-{fy_start_month}-01') + pd.DateOffset(months=-1)
fy_end_str = fy_end_dt.strftime('%b %Y') # e.g., Mar 2026

if clients_needed_from_new_leads <= 0 :
     st.success(f"✅ Target Met/Exceeded: Based on inputs, spillover, renewals, and pre-FY leads are projected to meet or exceed the net new target for the FY ending {fy_end_str}.")
     st.markdown(f"Revenue needed from *new leads generated this FY*: **₹{revenue_gap_for_new_leads/100000:.2f} L** ({clients_needed_from_new_leads} clients)")

elif not needs_met:
     st.error(f"⚠️ **Target Likely Unreachable Within FY ending {fy_end_str}**")
     st.markdown(f"Revenue needed from *new leads generated this FY*: **₹{revenue_gap_for_new_leads/100000:.2f} L** ({clients_needed_from_new_leads} clients)")
     if last_gen_month_index_for_fy <= 0 and avg_sales_cycle_months >= 12:
         st.markdown(f"Reason: With a **{avg_sales_cycle_months}-month** sales cycle, leads generated in Month 1 or later of this FY will close *next* financial year.")
     elif current_month_index > last_gen_month_index_for_fy :
         last_gen_month_str = (pd.Timestamp(f'2025-{fy_start_month}-01') + pd.DateOffset(months=last_gen_month_index_for_fy-1)).strftime('%b %Y')
         st.markdown(f"Reason: It is currently **Month {current_month_index}**. Leads needed to be generated by **Month {last_gen_month_index_for_fy} ({last_gen_month_str})** to close by {fy_end_str} (given the {avg_sales_cycle_months}-month cycle).")
     elif leads_per_deal == float('inf'):
          st.markdown("Reason: Cannot calculate lead requirements with a **0%** conversion rate.")
     else:
          st.markdown("Reason: Calculation based on current inputs indicates the timeline is too short or lead volume is too high.")

else: # needs_met is True and clients_needed_from_new_leads > 0
    col_a, col_b, col_c = st.columns(3)
    st.info(f"To close the remaining **₹{revenue_gap_for_new_leads/100000:.2f} L** ({clients_needed_from_new_leads} clients) *by {fy_end_str}*:")
    last_gen_month_str = (pd.Timestamp(f'2025-{fy_start_month}-01') + pd.DateOffset(months=last_gen_month_index_for_fy-1)).strftime('%b %Y')
    col_a.metric("Lead Gen Window Ends", f"Month {last_gen_month_index_for_fy} ({last_gen_month_str})", help=f"Leads must be generated by the end of this month to potentially close by {fy_end_str}.")
    col_b.metric(f"Avg. Monthly Leads Needed (Now - {last_gen_month_str})", f"{monthly_leads_needed_in_window:.0f}", help=f"Total {total_leads_needed_in_window:.0f} leads needed over the remaining {months_left_in_window} months of the window.")
    col_c.metric("Avg. Monthly Marketing Spend (During Window)", f"₹{monthly_spend_in_window:,.0f}", help=f"Based on CPL of ₹{cpl_rupees:,.0f}")


# --- Build Monthly Projection Table ---
st.header(f"🗓️ Monthly Revenue Projection (FY: {fy_start_month} 2025 - {fy_end_str})")
st.caption("This table projects when revenue might land based on lead generation month + sales cycle.")

# Generate month names based on FY Start
fy_start_dt = pd.Timestamp(f'2025-{fy_start_month}-01')
month_names = [(fy_start_dt + pd.DateOffset(months=i)).strftime('%b %Y') for i in range(12)]

# Initialize DataFrame
projection_df = pd.DataFrame({
    'Month': month_names,
    'Month Index': range(1, 13),
    'Leads Generated': np.zeros(12, dtype=int),
    'Monthly Spend (₹)': np.zeros(12, dtype=float), # Use float for spend calculation
    'Clients Closing': np.zeros(12, dtype=float),
    'Revenue Closing (₹)': np.zeros(12, dtype=float),
    'Cumulative Revenue (₹)': np.zeros(12, dtype=float)
})

# 1. Add Known Revenue (Spillover + Renewals) - Assume it lands early in Month 1
projection_df.loc[0, 'Revenue Closing (₹)'] += known_revenue
# Note: We will calculate cumulative at the end

# 2. Add Revenue from Pre-FY Leads (e.g., the 61 leads)
# Estimate closing months: Assume Feb/Mar '25 generation (~ -1/-2 months before FY start) + avg_sales_cycle_months
# Calculate index relative to FY start (0 = first month of FY)
approx_gen_month_offset = -2 # Average offset for Feb/Mar relative to Apr start (adjust if FY start isn't Apr)
if fy_start_month == 'Jan': approx_gen_month_offset = 1 # Feb/Mar relative to Jan start
elif fy_start_month == 'Jul': approx_gen_month_offset = -5 # Feb/Mar relative to Jul start
elif fy_start_month == 'Oct': approx_gen_month_offset = -8 # Feb/Mar relative to Oct start

closing_month_index_prior = approx_gen_month_offset + avg_sales_cycle_months # 0-based index relative to FY start

if 0 <= closing_month_index_prior < 12:
    # Add prior lead contribution to the calculated closing month
    projection_df.loc[closing_month_index_prior, 'Clients Closing'] += clients_from_prior_leads
    projection_df.loc[closing_month_index_prior, 'Revenue Closing (₹)'] += revenue_from_prior_leads
elif closing_month_index_prior >= 12 and leads_already_generated > 0:
     st.sidebar.warning(f"Pre-FY leads ({leads_already_generated}) might close *next* FY based on the {avg_sales_cycle_months}-month cycle.")
# else: closing_month_index_prior < 0 means they already closed last FY (no action needed here)


# 3. Add Leads Generated This FY and Their Projected Revenue
# Determine the actual generation months based on the window and current month
start_gen_month_idx = current_month_index - 1 # 0-based index for current month
end_gen_month_idx = last_gen_month_index_for_fy -1 # 0-based index for last gen month

if needs_met and clients_needed_from_new_leads > 0: # Only add leads if needed and possible
    for gen_month_idx in range(start_gen_month_idx, end_gen_month_idx + 1):
        if gen_month_idx < 0 or gen_month_idx >= 12: continue # Safety check

        projection_df.loc[gen_month_idx, 'Leads Generated'] = monthly_leads_needed_in_window
        projection_df.loc[gen_month_idx, 'Monthly Spend (₹)'] = monthly_spend_in_window

        # Calculate closing month index (0-based)
        closing_month_idx = gen_month_idx + avg_sales_cycle_months

        if closing_month_idx < 12: # Ensure it closes within this FY
            clients_closing_this_month = monthly_leads_needed_in_window * conversion_rate_decimal
            revenue_closing_this_month = clients_closing_this_month * avg_ticket_size
            projection_df.loc[closing_month_idx, 'Clients Closing'] += clients_closing_this_month
            projection_df.loc[closing_month_idx, 'Revenue Closing (₹)'] += revenue_closing_this_month
        # Else: these leads close next FY - not added to this FY's projection table


# 4. Add Sustainable Lead Gen for Future Pipeline (Optional Visualization)
# Placeholder: Add input later if needed for post-window lead gen visualization

# 5. Calculate Cumulative Revenue
projection_df['Cumulative Revenue (₹)'] = projection_df['Revenue Closing (₹)'].cumsum()


# --- Display Table and Chart ---

# Format the DataFrame for display
display_df = projection_df.copy()
# Format non-zero spend only, handle potential inf values gracefully
display_df['Monthly Spend (₹)'] = display_df['Monthly Spend (₹)'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) and x != float('inf') and x > 0 else "₹0")
display_df['Revenue Closing (₹)'] = display_df['Revenue Closing (₹)'].map('₹{:,.0f}'.format)
display_df['Cumulative Revenue (₹)'] = display_df['Cumulative Revenue (₹)'].map('₹{:,.0f}'.format)
display_df['Clients Closing'] = display_df['Clients Closing'].map('{:.1f}'.format) # Show decimals for avg
display_df['Leads Generated'] = display_df['Leads Generated'].apply(lambda x: f"{x:.0f}" if pd.notna(x) and x != float('inf') else "N/A")


st.dataframe(display_df[['Month', 'Leads Generated', 'Monthly Spend (₹)', 'Clients Closing', 'Revenue Closing (₹)', 'Cumulative Revenue (₹)']], use_container_width=True)

# Add a chart - Ensure 'Cumulative Revenue (₹)' is numeric before plotting
chart_data = projection_df[['Month', 'Cumulative Revenue (₹)']].copy()
chart_data['Target'] = total_revenue_target # Add target line
chart_data.set_index('Month', inplace=True)

st.line_chart(chart_data)


st.markdown("---")
st.subheader("Assumptions & Notes:")
# Determine prior lead closing month string for notes
prior_lead_closing_month_str = 'N/A (Check Inputs)'
if 0 <= closing_month_index_prior < 12:
    prior_lead_closing_month_str = f"around Month {closing_month_index_prior + 1} ({month_names[closing_month_index_prior]})"
elif closing_month_index_prior >= 12:
    prior_lead_closing_month_str = "next FY (projected)"
elif closing_month_index_prior < 0:
     prior_lead_closing_month_str = "last FY (projected)"


st.markdown(f"""
*   Financial Year starts **{fy_start_month} 2025** and ends **{fy_end_str}**.
*   The **{lead_to_deal_conversion_rate_percent}%** Lead-to-Deal conversion rate is applied uniformly. **This is a critical assumption - validate it!**
*   The **{avg_sales_cycle_months}-month** sales cycle is an average; actual deal closures will vary. Revenue is projected based on `Lead Gen Month + {avg_sales_cycle_months} months`.
*   Revenue is booked in the month the deal is projected to close.
*   Lead generation and CPL are assumed constant during the required generation window (Months {current_month_index} to {last_gen_month_index_for_fy}, if applicable and needed).
*   Spillover (**₹{spillover_revenue/100000:.1f}L**) & Renewal (**₹{renewals_revenue/100000:.1f}L**) revenue is assumed to land in Month 1 ({month_names[0]}).
*   Revenue from *pre-FY leads* ({leads_already_generated} leads, generating ~₹{revenue_from_prior_leads/100000:.1f}L) is projected to close {prior_lead_closing_month_str}.
*   The calculator focuses on leads needed to hit the **₹{total_revenue_target/100000:.0f}L** target *within this FY*. Sustainable lead generation beyond Month {last_gen_month_index_for_fy} is needed for future pipeline but not explicitly calculated here for *this* FY's revenue.
""")