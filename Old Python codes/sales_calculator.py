import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime

# --- Configuration and Page Title ---
st.set_page_config(layout="wide", page_title="SaaS Financial Forecaster")
st.title("📈 SaaS Sales, Expense & Profitability Forecaster")
st.caption("Adjust inputs in the sidebar to see financial projections and SaaS metrics change.")

# --- Helper function to get 0-based month index ---
def get_month_index(target_month_name, fy_start_month_name):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    try:
        target_idx_abs = months.index(target_month_name[:3].title())
        start_idx_abs = months.index(fy_start_month_name[:3].title())
        if target_idx_abs >= start_idx_abs:
            return target_idx_abs - start_idx_abs
        else:
            return (12 - start_idx_abs) + target_idx_abs
    except ValueError:
        st.error(f"Error: Invalid month name used: {target_month_name} or {fy_start_month_name}")
        return -1

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Strategy & Financial Year")

# --- Strategy Selection ---
planning_strategy = st.sidebar.radio(
    "Select Planning Strategy",
    ["Set Revenue Target", "Set Marketing Budget"],
    index=0,
    help="Choose whether to calculate required spend for a revenue goal OR project revenue based on a fixed marketing budget."
)

fy_start_month = st.sidebar.selectbox("Financial Year Start Month", ["Apr", "Jan", "Jul", "Oct"], index=0)
# Generate initial Timestamp for formatting function based on selected FY month
fy_start_dt_sidebar = pd.Timestamp(f'2025-{fy_start_month}-01')
current_month_index = st.sidebar.selectbox(
    "Select Current Month (of FY)", options=list(range(1, 13)),
    format_func=lambda x: (fy_start_dt_sidebar + pd.DateOffset(months=x-1)).strftime('%b %Y'), index=0
)

st.sidebar.header("🎯 Targets & Actuals")
total_revenue_target_lakhs = st.sidebar.number_input(
    "Overall Annual Revenue Target (Lakhs)", min_value=0, value=400, step=50,
    help="The ultimate goal for the full financial year."
)
actual_revenue_booked_lakhs = st.sidebar.number_input(
    "Actual Revenue Booked So Far (Lakhs)", min_value=0.0, value=0.0, step=5.0,
    help="Revenue already secured/invoiced this FY, excluding expected spillover/renewals."
)

# --- Conditional Input for Marketing Budget ---
total_planned_marketing_spend_lakhs = 0.0 # Default
if planning_strategy == "Set Marketing Budget":
    total_planned_marketing_spend_lakhs = st.sidebar.number_input(
        "Total Planned Marketing Spend for FY (Lakhs)", min_value=0.0, value=30.0, step=1.0,
        help="Enter the total marketing budget you intend to spend from the current month onwards."
    )

st.sidebar.header("💰 Revenue Details")
avg_ticket_size_lakhs = st.sidebar.number_input("Avg. Ticket Size (New Business) (Lakhs)", min_value=1.0, value=10.0, step=0.5)
spillover_total_lakhs = st.sidebar.number_input("Expected Spillover Revenue this FY (Lakhs)", min_value=0.0, value=70.0, step=5.0)
renewals_total_lakhs = st.sidebar.number_input("Expected Renewal Revenue this FY (Lakhs)", min_value=0.0, value=30.0, step=5.0)
renewal_client_count = st.sidebar.number_input("Number of Renewal Clients", min_value=0, value=10, step=1)
st.sidebar.caption(f"Implied renewal value: ₹{renewals_total_lakhs / renewal_client_count if renewal_client_count > 0 else 0:.2f} L/client.")

# --- LTV Inputs ---
st.sidebar.header("📈 Customer Lifetime Value (LTV) Inputs")
avg_amc_per_client_lakhs = st.sidebar.number_input("Avg. Annual Maintenance Contract (AMC) / Client (Lakhs)", min_value=0.0, value=4.0, step=0.5)
avg_phase2_value_lakhs = st.sidebar.number_input("Avg. Phase 2 / Expansion Value / Client (Lakhs)", min_value=0.0, value=10.0, step=0.5)
avg_client_lifespan_years = st.sidebar.number_input("Avg. Client Lifespan (Years)", min_value=1, value=3, step=1)

# --- Sales Cycle, Leads, Expenses, Headcount ---
# ... (These sidebar sections remain unchanged) ...
st.sidebar.header("⏳ Sales Cycle (New Business)")
avg_sales_cycle_months = st.sidebar.slider("Average Sales Cycle (Months)", min_value=3, max_value=18, value=9, step=1)

st.sidebar.header("📊 Lead & Conversion Metrics")
lead_to_deal_conversion_rate_percent = st.sidebar.slider("Overall Lead-to-Deal Conversion Rate (%)", min_value=0.1, max_value=15.0, value=3.2, step=0.1)
base_cpl_rupees = st.sidebar.number_input("Base Cost Per Lead (CPL) (₹)", min_value=100, value=3000, step=100)
cpl_increase_percent = st.sidebar.slider("CPL Increase / Escalation (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.5)
st.sidebar.markdown("---")
leads_already_generated = st.sidebar.number_input("Leads Generated *Before* This FY Start (Estimate)", min_value=0, value=61, step=1)

st.sidebar.header("💸 Monthly Operating Expenses")
monthly_sales_salary_lakhs = st.sidebar.number_input("Avg. Monthly Sales Salary Cost (Lakhs)", min_value=0.0, value=2.0, step=0.1)
monthly_product_salary_lakhs = st.sidebar.number_input("Avg. Monthly Product Salary Cost (Lakhs)", min_value=0.0, value=2.5, step=0.1)
monthly_services_salary_lakhs = st.sidebar.number_input("Avg. Monthly Services/Support Salary Cost (Lakhs)", min_value=0.0, value=1.5, step=0.1)
monthly_server_lakhs = st.sidebar.number_input("Avg. Monthly Server/Infra Cost (Lakhs)", min_value=0.0, value=1.0, step=0.1)
st.sidebar.caption(f"Total Fixed Monthly Salary: ₹{monthly_sales_salary_lakhs + monthly_product_salary_lakhs + monthly_services_salary_lakhs:.1f} Lakhs")

st.sidebar.header("👥 Headcount (Average for FY)")
avg_sales_headcount = st.sidebar.number_input("Avg. Sales Headcount", min_value=0, value=3, step=1)
avg_product_headcount = st.sidebar.number_input("Avg. Product Headcount", min_value=0, value=4, step=1)
avg_services_headcount = st.sidebar.number_input("Avg. Services/Support Headcount", min_value=0, value=2, step=1)
avg_other_headcount = st.sidebar.number_input("Avg. Other Headcount (G&A, etc.)", min_value=0, value=1, step=1)
total_headcount = avg_sales_headcount + avg_product_headcount + avg_services_headcount + avg_other_headcount
st.sidebar.caption(f"Total Avg. Headcount: {total_headcount}")

# --- Core Calculations ---

# --- Value Conversions ---
total_revenue_target = total_revenue_target_lakhs * 100000
actual_revenue_booked = actual_revenue_booked_lakhs * 100000
total_planned_marketing_spend = total_planned_marketing_spend_lakhs * 100000
avg_ticket_size = avg_ticket_size_lakhs * 100000
spillover_total = spillover_total_lakhs * 100000
renewals_total = renewals_total_lakhs * 100000
avg_amc_per_client = avg_amc_per_client_lakhs * 100000
avg_phase2_value = avg_phase2_value_lakhs * 100000
monthly_sales_salary_cost = monthly_sales_salary_lakhs * 100000
monthly_product_salary_cost = monthly_product_salary_lakhs * 100000
monthly_services_salary_cost = monthly_services_salary_lakhs * 100000
monthly_server_cost = monthly_server_lakhs * 100000

# --- LTV Calculation ---
ltv_per_customer = (avg_amc_per_client * avg_client_lifespan_years) + avg_phase2_value

# --- Lead Gen & Marketing Cost Basics ---
actual_cpl = base_cpl_rupees * (1 + cpl_increase_percent / 100)
conversion_rate_decimal = lead_to_deal_conversion_rate_percent / 100 if lead_to_deal_conversion_rate_percent > 0 else 0
leads_per_deal = math.ceil(1 / conversion_rate_decimal) if conversion_rate_decimal > 0 else float('inf')

# --- Revenue Timing ---
spillover_may_lakhs = 45.0
spillover_may = spillover_may_lakhs * 100000
spillover_jun_jul_total = max(0, spillover_total - spillover_may)
spillover_jun = spillover_jun_jul_total / 2
spillover_jul = spillover_jun_jul_total / 2

renewal_start_month = "Jul"
renewal_end_month = "Dec"
renewal_start_idx = get_month_index(renewal_start_month, fy_start_month)
renewal_end_idx = get_month_index(renewal_end_month, fy_start_month)
renewal_months_count = renewal_end_idx - renewal_start_idx + 1 if renewal_end_idx >= renewal_start_idx else 0
monthly_renewal_revenue = renewals_total / renewal_months_count if renewal_months_count > 0 else 0

# --- Calculate Initial Net New Target (Before Actuals) ---
initial_net_new_target = max(0, total_revenue_target - spillover_total - renewals_total)
# --- Calculate Remaining Target After Actuals ---
remaining_target_after_actuals = max(0, total_revenue_target - spillover_total - renewals_total - actual_revenue_booked)

# --- Calculate Revenue from Pre-FY Leads ---
revenue_from_prior_leads = leads_already_generated * conversion_rate_decimal * avg_ticket_size
clients_from_prior_leads = leads_already_generated * conversion_rate_decimal

# --- Declare variables needed for both strategies ---
monthly_leads_to_generate = 0
monthly_marketing_spend = 0
total_leads_needed_or_generated = 0
needs_met = True # Assume true initially

# --- Strategy-Based Calculations ---

start_gen_month_idx_num = current_month_index - 1 # 0-based index for current month

if planning_strategy == "Set Revenue Target":
    # --- Calculate effort needed to hit the REMAINING target ---
    revenue_gap_for_new_leads = max(0, remaining_target_after_actuals - revenue_from_prior_leads)
    clients_needed_from_new_leads = revenue_gap_for_new_leads / avg_ticket_size if avg_ticket_size > 0 else 0

    last_gen_month_index_for_fy = max(0, 12 - avg_sales_cycle_months) # 1-based index
    months_left_in_window = max(0, last_gen_month_index_for_fy - current_month_index + 1) if current_month_index <= last_gen_month_index_for_fy else 0

    if months_left_in_window > 0 and leads_per_deal != float('inf') and clients_needed_from_new_leads > 0:
        total_leads_needed_or_generated = math.ceil(clients_needed_from_new_leads * leads_per_deal)
        monthly_leads_to_generate = math.ceil(total_leads_needed_or_generated / months_left_in_window)
        monthly_marketing_spend = monthly_leads_to_generate * actual_cpl
        needs_met = True
    elif clients_needed_from_new_leads <= 0: # Target already met
        total_leads_needed_or_generated = 0
        monthly_leads_to_generate = 0
        monthly_marketing_spend = 0
        needs_met = True
    else: # Cannot meet target
        total_leads_needed_or_generated = float('inf') if clients_needed_from_new_leads > 0 else 0
        monthly_leads_to_generate = float('inf') if clients_needed_from_new_leads > 0 else 0
        monthly_marketing_spend = float('inf') if clients_needed_from_new_leads > 0 else 0
        needs_met = False
    # Define end generation month for population loop
    end_gen_month_idx_num = last_gen_month_index_for_fy - 1 # 0-based

elif planning_strategy == "Set Marketing Budget":
    months_to_spend = max(1, 12 - start_gen_month_idx_num) # Spend over remaining months (min 1)
    monthly_marketing_spend = total_planned_marketing_spend / months_to_spend if months_to_spend > 0 else 0
    monthly_leads_to_generate = monthly_marketing_spend / actual_cpl if actual_cpl > 0 else 0
    total_leads_needed_or_generated = monthly_leads_to_generate * months_to_spend # Total leads generated under budget
    # Define end generation month for population loop - generate leads till end of FY
    end_gen_month_idx_num = 11 # 0-based index for the last month (March if FY starts Apr)
    needs_met = True # We can always execute the plan, even if it doesn't hit the *original* target


# --- Build Monthly Projection Table ---
fy_start_dt_for_calc = pd.Timestamp(f'2025-{fy_start_month}-01')
month_names = []
month_timestamps = []
for i in range(12):
    current_month_start = fy_start_dt_for_calc + pd.DateOffset(months=i)
    month_names.append(current_month_start.strftime('%b %Y'))
    month_timestamps.append(current_month_start)

projection_df = pd.DataFrame({
    'Month': month_names,
    'Month_Timestamp': month_timestamps,
    # ... (rest of the columns are the same as before) ...
    'Month Index': range(1, 13),
    'Revenue: Spillover (₹)': np.zeros(12, dtype=float),
    'Revenue: Renewals (₹)': np.zeros(12, dtype=float),
    'Revenue: Pre-FY Leads (₹)': np.zeros(12, dtype=float),
    'Revenue: New Leads This FY (₹)': np.zeros(12, dtype=float),
    'TOTAL REVENUE (₹)': np.zeros(12, dtype=float),
    'Clients Closing: Pre-FY Leads': np.zeros(12, dtype=float),
    'Clients Closing: New Leads This FY': np.zeros(12, dtype=float),
    'Total New Clients Closing Monthly': np.zeros(12, dtype=float),
    'Leads Generated': np.zeros(12, dtype=float), # Use float for generated leads
    'Expense: Marketing (₹)': np.zeros(12, dtype=float),
    'Expense: Sales Salary (₹)': np.full(12, monthly_sales_salary_cost, dtype=float),
    'Expense: Product Salary (₹)': np.full(12, monthly_product_salary_cost, dtype=float),
    'Expense: Services Salary (₹)': np.full(12, monthly_services_salary_cost, dtype=float),
    'Expense: Server (₹)': np.full(12, monthly_server_cost, dtype=float),
    'TOTAL EXPENSES (₹)': np.zeros(12, dtype=float),
    'Monthly CAC Cost Base (₹)': np.zeros(12, dtype=float),
    'Monthly CAC (₹)': np.zeros(12, dtype=float),
    'Monthly COGS (₹)': np.zeros(12, dtype=float),
    'Monthly Gross Profit (₹)': np.zeros(12, dtype=float),
    'Monthly Gross Profit Margin (%)': np.zeros(12, dtype=float),
    'Monthly Product Exp % Revenue (%)': np.zeros(12, dtype=float),
    'Monthly P/L (₹)': np.zeros(12, dtype=float),
    'Cumulative P/L (₹)': np.zeros(12, dtype=float)
})

# --- Populate DataFrame Step-by-Step (using 0-11 index) ---

# 1. Revenue Streams (Spillover, Renewals, Pre-FY Leads) - This logic is independent of strategy
# ... (Code for populating these remains the same) ...
may_idx = get_month_index("May", fy_start_month)
jun_idx = get_month_index("Jun", fy_start_month)
jul_idx = get_month_index("Jul", fy_start_month)
if 0 <= may_idx < 12: projection_df.loc[may_idx, 'Revenue: Spillover (₹)'] += spillover_may
if 0 <= jun_idx < 12: projection_df.loc[jun_idx, 'Revenue: Spillover (₹)'] += spillover_jun
if 0 <= jul_idx < 12: projection_df.loc[jul_idx, 'Revenue: Spillover (₹)'] += spillover_jul

if renewal_months_count > 0:
    for month_offset in range(renewal_months_count):
        current_renewal_month_idx = renewal_start_idx + month_offset
        if 0 <= current_renewal_month_idx < 12:
            projection_df.loc[current_renewal_month_idx, 'Revenue: Renewals (₹)'] += monthly_renewal_revenue

approx_gen_month_offset = -2 # Relative to Apr
if fy_start_month == 'Jan': approx_gen_month_offset = 1
elif fy_start_month == 'Jul': approx_gen_month_offset = -5
elif fy_start_month == 'Oct': approx_gen_month_offset = -8
closing_month_index_prior = approx_gen_month_offset + avg_sales_cycle_months # 0-based relative to FY start

if 0 <= closing_month_index_prior < 12:
    projection_df.loc[closing_month_index_prior, 'Revenue: Pre-FY Leads (₹)'] += revenue_from_prior_leads
    projection_df.loc[closing_month_index_prior, 'Clients Closing: Pre-FY Leads'] += clients_from_prior_leads
elif closing_month_index_prior >= 12 and leads_already_generated > 0:
    st.sidebar.warning(f"Pre-FY leads ({leads_already_generated}) might close *next* FY.")

# 2. Populate Leads Generated, Marketing Costs based on STRATEGY
# Also calculate corresponding Revenue from New Leads This FY
if needs_met and monthly_leads_to_generate != float('inf'):
    for gen_month_idx in range(start_gen_month_idx_num, end_gen_month_idx_num + 1):
        if gen_month_idx < 0 or gen_month_idx >= 12: continue

        # Populate based on calculated monthly values (either required or planned)
        projection_df.loc[gen_month_idx, 'Leads Generated'] = monthly_leads_to_generate
        projection_df.loc[gen_month_idx, 'Expense: Marketing (₹)'] = monthly_marketing_spend

        # Calculate resulting revenue based on these leads
        closing_month_idx = gen_month_idx + avg_sales_cycle_months # 0-based

        if closing_month_idx < 12: # Closes within this FY
            # Use the leads generated for THIS month
            clients_closing_this_month_batch = projection_df.loc[gen_month_idx, 'Leads Generated'] * conversion_rate_decimal
            revenue_closing_this_month_batch = clients_closing_this_month_batch * avg_ticket_size

            projection_df.loc[closing_month_idx, 'Revenue: New Leads This FY (₹)'] += revenue_closing_this_month_batch
            projection_df.loc[closing_month_idx, 'Clients Closing: New Leads This FY'] += clients_closing_this_month_batch

# 3. Calculate Intermediate & Total Columns
# ... (This calculation block remains the same as before, using the populated data) ...
projection_df['Total New Clients Closing Monthly'] = projection_df['Clients Closing: Pre-FY Leads'] + projection_df['Clients Closing: New Leads This FY']
projection_df['TOTAL REVENUE (₹)'] = projection_df[[
    'Revenue: Spillover (₹)', 'Revenue: Renewals (₹)', 'Revenue: Pre-FY Leads (₹)', 'Revenue: New Leads This FY (₹)'
]].sum(axis=1)
projection_df['TOTAL EXPENSES (₹)'] = projection_df[[
    'Expense: Marketing (₹)', 'Expense: Sales Salary (₹)', 'Expense: Product Salary (₹)',
    'Expense: Services Salary (₹)', 'Expense: Server (₹)'
]].sum(axis=1)
projection_df['Monthly CAC Cost Base (₹)'] = projection_df['Expense: Marketing (₹)'] + projection_df['Expense: Sales Salary (₹)']
projection_df['Monthly CAC (₹)'] = projection_df['Monthly CAC Cost Base (₹)'].div(projection_df['Total New Clients Closing Monthly']).replace([np.inf, -np.inf], 0).fillna(0)
projection_df['Monthly COGS (₹)'] = projection_df['Expense: Services Salary (₹)'] + projection_df['Expense: Server (₹)']
projection_df['Monthly Gross Profit (₹)'] = projection_df['TOTAL REVENUE (₹)'] - projection_df['Monthly COGS (₹)']
projection_df['Monthly Gross Profit Margin (%)'] = projection_df['Monthly Gross Profit (₹)'].div(projection_df['TOTAL REVENUE (₹)']).replace([np.inf, -np.inf], 0).fillna(0) * 100
projection_df['Monthly Product Exp % Revenue (%)'] = projection_df['Expense: Product Salary (₹)'].div(projection_df['TOTAL REVENUE (₹)']).replace([np.inf, -np.inf], 0).fillna(0) * 100
projection_df['Monthly P/L (₹)'] = projection_df['TOTAL REVENUE (₹)'] - projection_df['TOTAL EXPENSES (₹)']
projection_df['Cumulative P/L (₹)'] = projection_df['Monthly P/L (₹)'].cumsum()


# --- Display KPIs ---
st.header("📊 Key Performance Indicators (KPIs)")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
kpi_col1.metric("Overall Revenue Target (FY)", f"₹{total_revenue_target/100000:.1f} L")
# Calculate remaining target considering ALL sources vs goal
projected_total_fy_revenue_interim = projection_df['TOTAL REVENUE (₹)'].sum() # interim calculation
target_remaining_display = max(0, total_revenue_target - actual_revenue_booked - projected_total_fy_revenue_interim) # What's left vs Goal after projection
kpi_col2.metric("Target Remaining (vs Goal)", f"₹{target_remaining_display/100000:.1f} L", help="Goal Target - Actual Booked - Projected Revenue for rest of FY")
kpi_col3.metric("Leads Needed Per Deal", f"{leads_per_deal:.0f}", help=f"Based on {lead_to_deal_conversion_rate_percent}% conversion")
kpi_col4.metric("Adjusted CPL", f"₹{actual_cpl:,.0f}", help=f"{cpl_increase_percent:.1f}% increase on base ₹{base_cpl_rupees:,.0f}")

st.markdown("---")

# --- Conditional Display: Lead Gen Requirements or Marketing Plan ---
if planning_strategy == "Set Revenue Target":
    st.header("🎯 Lead Generation Required (To Hit Remaining Target)")
    # ... (Error/Success message logic remains the same, but based on `remaining_target_after_actuals`) ...
    fy_end_dt = fy_start_dt_for_calc + pd.DateOffset(years=1) + pd.DateOffset(days=-1)
    fy_end_str = fy_end_dt.strftime('%b %Y')
    target_to_calculate_from = remaining_target_after_actuals # Use the target remaining after booked revenue
    clients_needed_display = math.ceil(max(0, target_to_calculate_from - revenue_from_prior_leads) / avg_ticket_size if avg_ticket_size > 0 else 0)
    revenue_gap_display = max(0, target_to_calculate_from - revenue_from_prior_leads)

    if clients_needed_display <= 0 :
        st.success(f"✅ Target Potentially Met/Exceeded: Based on inputs, actuals, spillover, renewals, and pre-FY leads are projected to meet or exceed the *remaining target* (₹{target_to_calculate_from / 100000:.1f} L) for the FY ending {fy_end_str}.")
        st.markdown(f"Revenue needed from *new leads generated this FY*: **₹{revenue_gap_display/100000:.2f} L** ({clients_needed_display} clients)")

    elif not needs_met: # Needs_met was set based on the gap calculation
        st.error(f"⚠️ **Target Likely Unreachable Within FY ending {fy_end_str}**")
        st.markdown(f"Remaining revenue needed from *new leads generated this FY*: **₹{revenue_gap_display/100000:.2f} L** ({clients_needed_display} clients)")
        # ... (Reasons remain the same) ...
        if last_gen_month_index_for_fy <= 0 and avg_sales_cycle_months >= 12:
            st.markdown(f"Reason: With a **{avg_sales_cycle_months}-month** sales cycle, leads generated in Month 1 or later of this FY will close *next* financial year.")
        elif current_month_index > last_gen_month_index_for_fy :
            last_gen_month_str = (fy_start_dt_for_calc + pd.DateOffset(months=last_gen_month_index_for_fy-1)).strftime('%b %Y') if last_gen_month_index_for_fy > 0 else "N/A"
            st.markdown(f"Reason: It is currently **Month {current_month_index}**. Leads needed to be generated by **Month {last_gen_month_index_for_fy} ({last_gen_month_str})** to close by {fy_end_str} (given the {avg_sales_cycle_months}-month cycle).")
        elif leads_per_deal == float('inf'):
            st.markdown("Reason: Cannot calculate lead requirements with a **0%** conversion rate.")
        else:
            st.markdown("Reason: Calculation based on current inputs indicates the timeline is too short or lead volume is too high.")

    else: # Needs_met is True and clients needed > 0
        req_col_a, req_col_b, req_col_c = st.columns(3)
        st.info(f"To close the remaining new business gap of **₹{revenue_gap_display/100000:.2f} L** ({clients_needed_display} clients) *by {fy_end_str}*:")
        last_gen_month_str = (fy_start_dt_for_calc + pd.DateOffset(months=last_gen_month_index_for_fy-1)).strftime('%b %Y') if last_gen_month_index_for_fy > 0 else "N/A"
        req_col_a.metric("Lead Gen Window Ends", f"Month {last_gen_month_index_for_fy} ({last_gen_month_str})")
        req_col_b.metric(f"Avg. Monthly Leads Needed (Now - {last_gen_month_str})", f"{monthly_leads_to_generate:.0f}", help=f"Total {total_leads_needed_or_generated:.0f} leads needed over the remaining {months_left_in_window} months of the window.")
        req_col_c.metric("Avg. Monthly Marketing Spend (During Window)", f"₹{monthly_marketing_spend:,.0f}", help=f"Based on Adjusted CPL of ₹{actual_cpl:,.0f}")

elif planning_strategy == "Set Marketing Budget":
    st.header("📈 Marketing Plan & Revenue Projection")
    plan_col_a, plan_col_b, plan_col_c = st.columns(3)
    plan_col_a.metric("Total Planned Marketing Spend", f"₹{total_planned_marketing_spend/100000:.1f} L")
    plan_col_b.metric("Avg. Monthly Marketing Spend (Planned)", f"₹{monthly_marketing_spend:,.0f}", help=f"Spread over {max(1, 12 - start_gen_month_idx_num)} remaining months")
    plan_col_c.metric("Avg. Monthly Leads Generated (Planned)", f"{monthly_leads_to_generate:.1f}", help=f"Based on planned spend and CPL of ₹{actual_cpl:,.0f}")


# --- Calculate Final FY KPIs & Metrics ---
# (Use the data already in projection_df, no need to reset index here)
total_fy_revenue = projection_df['TOTAL REVENUE (₹)'].sum() + actual_revenue_booked # Add booked actuals for final total
total_fy_expenses = projection_df['TOTAL EXPENSES (₹)'].sum()
total_fy_pl = total_fy_revenue - total_fy_expenses # Recalculate P/L with actuals included
target_delta = total_fy_revenue - total_revenue_target

total_fy_new_clients = projection_df['Total New Clients Closing Monthly'].sum()
total_fy_mktg_spend = projection_df['Expense: Marketing (₹)'].sum()
total_fy_sales_salary = projection_df['Expense: Sales Salary (₹)'].sum()
total_fy_product_salary = projection_df['Expense: Product Salary (₹)'].sum()
total_fy_cogs = projection_df['Monthly COGS (₹)'].sum()
# Recalculate Gross Profit based on total revenue including actuals
total_fy_gross_profit = total_fy_revenue - total_fy_cogs

overall_fy_cac = (total_fy_mktg_spend + total_fy_sales_salary) / total_fy_new_clients if total_fy_new_clients > 0 else 0
overall_fy_gross_profit_margin = (total_fy_gross_profit / total_fy_revenue) * 100 if total_fy_revenue > 0 else 0
overall_fy_product_exp_perc = (total_fy_product_salary / total_fy_revenue) * 100 if total_fy_revenue > 0 else 0
overall_fy_rev_per_employee = total_fy_revenue / total_headcount if total_headcount > 0 else 0
ltv_cac_ratio = ltv_per_customer / overall_fy_cac if overall_fy_cac > 0 else 0


# --- Display Final FY Summary & SaaS Metrics ---
st.markdown("---")
st.header("🏁 Projected Financial Year Summary (Incl. Actuals Booked)")
final_kpi1, final_kpi2, final_kpi3 = st.columns(3)
final_kpi1.metric("Projected Total Revenue (Incl. Actuals)", f"₹{total_fy_revenue/100000:.1f} L", f"₹{target_delta/100000:.1f} L vs Goal")
final_kpi2.metric("Projected Total Expenses", f"₹{total_fy_expenses/100000:.1f} L")
final_kpi3.metric("Projected Profit/Loss", f"₹{total_fy_pl/100000:.1f} L")

# --- SaaS Metrics Display (remains the same) ---
st.subheader("Key SaaS Metrics (Projected for Full FY)")
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col4, metric_col5, metric_col6 = st.columns(3)

metric_col1.metric("LTV (Customer Lifetime Value)", f"₹{ltv_per_customer/100000:.1f} L", help="(Avg AMC/Yr * Lifespan Yrs) + Avg Phase 2 Value")
metric_col2.metric("CAC (Customer Acquisition Cost)", f"₹{overall_fy_cac:,.0f}", help="(Marketing Spend + Sales Salary) / New Clients Acquired")
metric_col3.metric("LTV : CAC Ratio", f"{ltv_cac_ratio:.1f} : 1", help="Ratio of Lifetime Value to Acquisition Cost")

metric_col4.metric("Gross Profit Margin", f"{overall_fy_gross_profit_margin:.1f}%", help="(Revenue - COGS [Services Sal + Infra]) / Revenue")
metric_col5.metric("Product Expense % Revenue", f"{overall_fy_product_exp_perc:.1f}%", help="(Product Salary / Revenue) * 100")
metric_col6.metric("Revenue Per Employee", f"₹{overall_fy_rev_per_employee/100000:.1f} L", help="Total Revenue / Total Avg Headcount")


# --- Display Monthly Projection Table ---
st.header(f"🗓️ Monthly Financial Projection (FY: {fy_start_month} 2025 - {fy_end_str})")
# Prepare DF for display
display_table_df = projection_df.copy()
display_table_df = display_table_df.reset_index(drop=True) # Use default index for table prep

cols_to_display = [
    'Month', 'TOTAL REVENUE (₹)', 'Total New Clients Closing Monthly', 'Monthly CAC (₹)',
    'Monthly Gross Profit Margin (%)', 'TOTAL EXPENSES (₹)', 'Monthly P/L (₹)',
    'Cumulative P/L (₹)', 'Leads Generated',
]
display_table_df = display_table_df[cols_to_display]

# Apply formatting
# (Formatting code remains the same)
display_table_df['TOTAL REVENUE (₹)'] = display_table_df['TOTAL REVENUE (₹)'].apply(lambda x: f"₹{x:,.0f}")
display_table_df['Total New Clients Closing Monthly'] = display_table_df['Total New Clients Closing Monthly'].apply(lambda x: f"{x:.1f}")
display_table_df['Monthly CAC (₹)'] = display_table_df['Monthly CAC (₹)'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) and x > 0 and x != float('inf') else "N/A")
display_table_df['Monthly Gross Profit Margin (%)'] = display_table_df['Monthly Gross Profit Margin (%)'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) and x != float('inf') else "N/A")
display_table_df['TOTAL EXPENSES (₹)'] = display_table_df['TOTAL EXPENSES (₹)'].apply(lambda x: f"₹{x:,.0f}")
display_table_df['Monthly P/L (₹)'] = display_table_df['Monthly P/L (₹)'].apply(lambda x: f"₹{x:,.0f}")
display_table_df['Cumulative P/L (₹)'] = display_table_df['Cumulative P/L (₹)'].apply(lambda x: f"₹{x:,.0f}")
# Use .0f for integer display of leads
display_table_df['Leads Generated'] = display_table_df['Leads Generated'].apply(lambda x: f"{x:.0f}" if pd.notna(x) and x != float('inf') else "N/A")


st.dataframe(display_table_df.set_index('Month'), use_container_width=True)

# --- Charts ---
st.markdown("---")
st.header("📊 Charts")
st.caption("Charts use the underlying datetime for x-axis ordering.")

# Prepare DF for plotting with DATETIME index
plot_df = projection_df.set_index('Month_Timestamp')

# (Charting code remains the same, using plot_df)
st.subheader("Revenue Breakdown")
revenue_cols = ['Revenue: Spillover (₹)', 'Revenue: Renewals (₹)', 'Revenue: Pre-FY Leads (₹)', 'Revenue: New Leads This FY (₹)']
st.bar_chart(plot_df[revenue_cols])

st.subheader("Profitability")
st.line_chart(plot_df[['Cumulative P/L (₹)', 'Monthly P/L (₹)']])

st.subheader("Expense Breakdown")
expense_cols = ['Expense: Marketing (₹)', 'Expense: Sales Salary (₹)', 'Expense: Product Salary (₹)', 'Expense: Services Salary (₹)', 'Expense: Server (₹)']
st.bar_chart(plot_df[expense_cols])

st.subheader("Customer Acquisition Cost (Monthly)")
cac_chart_data = plot_df[['Monthly CAC (₹)']].copy()
cac_chart_data = cac_chart_data[cac_chart_data['Monthly CAC (₹)'] > 0]
if not cac_chart_data.empty:
    st.line_chart(cac_chart_data)
else:
    st.caption("No new clients acquired in projected months to calculate Monthly CAC.")


# --- Notes ---
st.markdown("---")
st.subheader("Assumptions & Notes:")
# (Notes section remains the same - ensure variable names match if needed)
# Determine prior lead closing month string for notes
prior_lead_closing_month_str = 'N/A (Check Inputs)'
if 0 <= closing_month_index_prior < 12:
    prior_lead_closing_month_str = f"around Month {closing_month_index_prior + 1} ({(fy_start_dt_for_calc + pd.DateOffset(months=closing_month_index_prior)).strftime('%b %Y')})"
elif closing_month_index_prior >= 12:
    prior_lead_closing_month_str = "next FY (projected)"
elif closing_month_index_prior < 0:
     prior_lead_closing_month_str = "last FY (projected)"

# Calculate start and end month strings for lead gen window note
start_gen_month_str = (fy_start_dt_for_calc + pd.DateOffset(months=start_gen_month_idx_num)).strftime('%b %Y') if start_gen_month_idx_num >=0 else 'N/A'
# Adjust end month string based on strategy
if planning_strategy == "Set Revenue Target":
    end_gen_month_str = (fy_start_dt_for_calc + pd.DateOffset(months=end_gen_month_idx_num)).strftime('%b %Y') if end_gen_month_idx_num >=0 and last_gen_month_index_for_fy > 0 else 'N/A'
else: # Marketing Budget strategy runs till end of year
    end_gen_month_str = (fy_start_dt_for_calc + pd.DateOffset(months=11)).strftime('%b %Y') # Always last month of FY

renewal_start_str = (fy_start_dt_for_calc + pd.DateOffset(months=renewal_start_idx)).strftime('%b %Y') if renewal_start_idx >=0 else 'N/A'
renewal_end_str = (fy_start_dt_for_calc + pd.DateOffset(months=renewal_end_idx)).strftime('%b %Y') if renewal_end_idx >=0 else 'N/A'
may_month_str_note = (fy_start_dt_for_calc + pd.DateOffset(months=get_month_index("May", fy_start_month))).strftime('%b %Y')
jun_month_str_note = (fy_start_dt_for_calc + pd.DateOffset(months=get_month_index("Jun", fy_start_month))).strftime('%b %Y')
jul_month_str_note = (fy_start_dt_for_calc + pd.DateOffset(months=get_month_index("Jul", fy_start_month))).strftime('%b %Y')
current_month_str_note = (fy_start_dt_for_calc + pd.DateOffset(months=current_month_index-1)).strftime('%b %Y')

st.markdown(f"""
*   **Strategy:** Currently using **{planning_strategy}**.
*   Financial Year: **{fy_start_month} 2025** to **{fy_end_str}**. Current assumed month: **{current_month_str_note}**.
*   New Business: Avg. Ticket **₹{avg_ticket_size/100000:.1f}L**, Sales Cycle **{avg_sales_cycle_months} months**, Conversion **{lead_to_deal_conversion_rate_percent}%**.
*   Spillover lands: ₹{spillover_may/100000:.1f}L in {may_month_str_note}, ₹{spillover_jun/100000:.1f}L in {jun_month_str_note}, ₹{spillover_jul/100000:.1f}L in {jul_month_str_note}.
*   Renewals spread from **{renewal_start_str}** to **{renewal_end_str}**.
*   Revenue from Pre-FY Leads ({leads_already_generated} leads) projected to close {prior_lead_closing_month_str}.
*   **Lead Generation & Spend Calculation:**
    *   If **Set Revenue Target:** Calculated for months **{start_gen_month_str}** to **{end_gen_month_str}** to meet remaining goal (after actuals).
    *   If **Set Marketing Budget:** Planned spend of **₹{total_planned_marketing_spend/100000:.1f}L** distributed from **{start_gen_month_str}** to **{end_gen_month_str}**.
*   Adjusted CPL: **₹{actual_cpl:,.0f}**.
*   Monthly Fixed Costs: Sales Sal(₹{monthly_sales_salary_cost/100000:.1f}L), Prod Sal(₹{monthly_product_salary_cost/100000:.1f}L), Serv Sal(₹{monthly_services_salary_cost/100000:.1f}L), Server(₹{monthly_server_cost/100000:.1f}L).
*   **Actual Revenue Booked** (₹{actual_revenue_booked/100000:.1f}L) reduces the target needed from future activities in 'Revenue Target' mode and is added to the final FY revenue total.
*   **LTV** = (Avg AMC/Yr * Lifespan Yrs) + Avg Phase 2 Value = **₹{ltv_per_customer/100000:.1f} L**.
*   **CAC** = (Monthly Marketing Spend + Monthly Sales Salary) / (New Clients This Month). FY Avg: **₹{overall_fy_cac:,.0f}**.
*   **LTV:CAC Ratio** (FY Avg): **{ltv_cac_ratio:.1f} : 1**.
*   **COGS** = Monthly Services Salary + Monthly Server Cost.
*   **Gross Profit** = Total Revenue - COGS. FY Margin: **{overall_fy_gross_profit_margin:.1f}%**.
*   **Product Expense %** = Monthly Product Salary / Total Revenue. FY Avg: **{overall_fy_product_exp_perc:.1f}%**.
*   **Revenue Per Employee** (FY Avg): **₹{overall_fy_rev_per_employee/100000:.1f} L**.
*   All projections are estimates based on averages and timing assumptions. Actual results will vary.
""")