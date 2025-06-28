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
        if target_idx_abs >= start_idx_abs: return target_idx_abs - start_idx_abs
        else: return (12 - start_idx_abs) + target_idx_abs
    except ValueError: st.error(f"Invalid month: {target_month_name}/{fy_start_month_name}"); return -1

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Strategy & Financial Year")
planning_strategy = st.sidebar.radio("Select Planning Strategy", ["Set Revenue Target", "Set Marketing Budget"], index=0)
fy_start_month = st.sidebar.selectbox("Financial Year Start Month", ["Apr", "Jan", "Jul", "Oct"], index=0)
fy_start_dt_sidebar = pd.Timestamp(f'2025-{fy_start_month}-01')
current_month_index = st.sidebar.selectbox("Select Current Month (of FY)", options=list(range(1, 13)), format_func=lambda x: (fy_start_dt_sidebar + pd.DateOffset(months=x-1)).strftime('%b %Y'), index=0)

st.sidebar.header("🎯 Goal & Actuals")
total_revenue_target_lakhs = st.sidebar.number_input("Overall Annual Revenue Target (Lakhs)", min_value=0, value=400, step=50, help="The ultimate goal for the full FY.")
actual_revenue_booked_lakhs = st.sidebar.number_input("Actual Revenue Booked So Far (Lakhs)", min_value=0.0, value=0.0, step=5.0, help="Revenue already secured/invoiced this FY.")

total_planned_marketing_spend_lakhs = 0.0
if planning_strategy == "Set Marketing Budget":
    total_planned_marketing_spend_lakhs = st.sidebar.number_input("Total Planned Marketing Spend for Rest of FY (Lakhs)", min_value=0.0, value=30.0, step=1.0)

st.sidebar.header("💰 Expected Revenue Details (Future)")
avg_ticket_size_lakhs = st.sidebar.number_input("Avg. Ticket Size (New Business) (Lakhs)", min_value=1.0, value=10.0, step=0.5)
spillover_total_lakhs = st.sidebar.number_input("Expected Spillover Revenue this FY (Lakhs)", min_value=0.0, value=70.0, step=5.0)
renewals_total_lakhs = st.sidebar.number_input("Expected Renewal Revenue this FY (Lakhs)", min_value=0.0, value=30.0, step=5.0)
renewal_client_count = st.sidebar.number_input("Number of Renewal Clients", min_value=0, value=10, step=1)
st.sidebar.caption(f"Implied renewal value: ₹{renewals_total_lakhs / renewal_client_count if renewal_client_count > 0 else 0:.2f} L/client.")

st.sidebar.header("📈 Customer Lifetime Value (LTV) Inputs")
avg_amc_per_client_lakhs = st.sidebar.number_input("Avg. Annual Maintenance Contract (AMC) / Client (Lakhs)", min_value=0.0, value=4.0, step=0.5)
avg_phase2_value_lakhs = st.sidebar.number_input("Avg. Phase 2 / Expansion Value / Client (Lakhs)", min_value=0.0, value=10.0, step=0.5)
avg_client_lifespan_years = st.sidebar.number_input("Avg. Client Lifespan (Years)", min_value=1, value=3, step=1)

# ... (Sales Cycle, Leads, Expenses, Headcount sidebar inputs remain the same) ...
st.sidebar.header("⏳ Sales Cycle (New Business)")
avg_sales_cycle_months = st.sidebar.slider("Average Sales Cycle (Months)", min_value=3, max_value=18, value=9, step=1)
st.sidebar.header("📊 Lead & Conversion Metrics")
lead_to_deal_conversion_rate_percent = st.sidebar.slider("Overall Lead-to-Deal Conversion Rate (%)", min_value=0.1, max_value=15.0, value=3.2, step=0.1)
base_cpl_rupees = st.sidebar.number_input("Base Cost Per Lead (CPL) (₹)", min_value=100, value=3000, step=100)
cpl_increase_percent = st.sidebar.slider("CPL Increase / Escalation (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.5)
st.sidebar.markdown("---"); leads_already_generated = st.sidebar.number_input("Leads Generated *Before* This FY Start (Estimate)", min_value=0, value=61, step=1)
st.sidebar.header("💸 Monthly Operating Expenses")
monthly_sales_salary_lakhs = st.sidebar.number_input("Avg. Monthly Sales Salary Cost (Lakhs)", min_value=0.0, value=2.0, step=0.1)
monthly_product_salary_lakhs = st.sidebar.number_input("Avg. Monthly Product Salary Cost (Lakhs)", min_value=0.0, value=2.5, step=0.1)
monthly_services_salary_lakhs = st.sidebar.number_input("Avg. Monthly Services/Support Salary Cost (Lakhs)", min_value=0.0, value=1.5, step=0.1)
monthly_server_lakhs = st.sidebar.number_input("Avg. Monthly Server/Infra Cost (Lakhs)", min_value=0.0, value=1.0, step=0.1)
st.sidebar.caption(f"Total Fixed Monthly Salary: ₹{monthly_sales_salary_lakhs + monthly_product_salary_lakhs + monthly_services_salary_lakhs:.1f} Lakhs")
st.sidebar.header("👥 Headcount (Average for FY)")
avg_sales_headcount = st.sidebar.number_input("Avg. Sales Headcount", min_value=0, value=3, step=1); avg_product_headcount = st.sidebar.number_input("Avg. Product Headcount", min_value=0, value=4, step=1)
avg_services_headcount = st.sidebar.number_input("Avg. Services/Support Headcount", min_value=0, value=2, step=1); avg_other_headcount = st.sidebar.number_input("Avg. Other Headcount (G&A, etc.)", min_value=0, value=1, step=1)
total_headcount = avg_sales_headcount + avg_product_headcount + avg_services_headcount + avg_other_headcount
st.sidebar.caption(f"Total Avg. Headcount: {total_headcount}")

# --- Core Calculations ---

# --- Value Conversions ---
total_revenue_target = total_revenue_target_lakhs * 100000 # The GOAL
actual_revenue_booked = actual_revenue_booked_lakhs * 100000 # What's DONE
total_planned_marketing_spend = total_planned_marketing_spend_lakhs * 100000
avg_ticket_size = avg_ticket_size_lakhs * 100000
spillover_total = spillover_total_lakhs * 100000 # Expected future
renewals_total = renewals_total_lakhs * 100000 # Expected future
avg_amc_per_client = avg_amc_per_client_lakhs * 100000
avg_phase2_value = avg_phase2_value_lakhs * 100000
monthly_sales_salary_cost = monthly_sales_salary_lakhs * 100000; monthly_product_salary_cost = monthly_product_salary_lakhs * 100000
monthly_services_salary_cost = monthly_services_salary_lakhs * 100000; monthly_server_cost = monthly_server_lakhs * 100000

# --- FY Dates ---
fy_start_dt_for_calc = pd.Timestamp(f'2025-{fy_start_month}-01')
fy_end_dt = fy_start_dt_for_calc + pd.DateOffset(years=1) + pd.DateOffset(days=-1)
fy_end_str = fy_end_dt.strftime('%b %Y')

# --- LTV Calculation ---
ltv_per_customer = (avg_amc_per_client * avg_client_lifespan_years) + avg_phase2_value

# --- Lead Gen & Marketing Cost Basics ---
actual_cpl = base_cpl_rupees * (1 + cpl_increase_percent / 100)
conversion_rate_decimal = lead_to_deal_conversion_rate_percent / 100 if lead_to_deal_conversion_rate_percent > 0 else 0
leads_per_deal = math.ceil(1 / conversion_rate_decimal) if conversion_rate_decimal > 0 else float('inf')

# --- Revenue Timing ---
spillover_may_lakhs = 45.0; spillover_may = spillover_may_lakhs * 100000
spillover_jun_jul_total = max(0, spillover_total - spillover_may); spillover_jun = spillover_jun_jul_total / 2; spillover_jul = spillover_jun_jul_total / 2
renewal_start_month = "Jul"; renewal_end_month = "Dec"
renewal_start_idx = get_month_index(renewal_start_month, fy_start_month); renewal_end_idx = get_month_index(renewal_end_month, fy_start_month)
renewal_months_count = renewal_end_idx - renewal_start_idx + 1 if renewal_end_idx >= renewal_start_idx else 0
monthly_renewal_revenue = renewals_total / renewal_months_count if renewal_months_count > 0 else 0

# --- Calculate Revenue Expected from Pre-FY Leads ---
revenue_from_prior_leads = leads_already_generated * conversion_rate_decimal * avg_ticket_size
clients_from_prior_leads = leads_already_generated * conversion_rate_decimal

# --- Calculate Target Remaining (for KPI) ---
target_remaining_kpi = max(0, total_revenue_target - actual_revenue_booked)

# --- Calculate Revenue needed *specifically from new leads* this FY to hit the Overall Target ---
# This is used ONLY by the "Set Revenue Target" strategy's calculation logic
revenue_needed_from_new_leads_to_hit_goal = max(0, total_revenue_target - actual_revenue_booked - spillover_total - renewals_total - revenue_from_prior_leads)

# --- Declare variables for monthly effort/outcome ---
monthly_leads_to_generate = 0
monthly_marketing_spend = 0
total_leads_needed_or_generated = 0
needs_met = True
start_gen_month_idx_num = current_month_index - 1
end_gen_month_idx_num = 11 # Default to end of FY

# --- Strategy-Based Calculations for Monthly Effort/Outcome ---
if planning_strategy == "Set Revenue Target":
    # Calculate effort needed to generate 'revenue_needed_from_new_leads_to_hit_goal'
    clients_needed_from_new_leads = revenue_needed_from_new_leads_to_hit_goal / avg_ticket_size if avg_ticket_size > 0 else 0

    last_gen_month_index_for_fy = max(0, 12 - avg_sales_cycle_months) # 1-based
    months_left_in_window = max(0, last_gen_month_index_for_fy - current_month_index + 1) if current_month_index <= last_gen_month_index_for_fy else 0
    end_gen_month_idx_num = last_gen_month_index_for_fy - 1 # Adjust end month for this strategy

    if months_left_in_window > 0 and leads_per_deal != float('inf') and clients_needed_from_new_leads > 0:
        total_leads_needed_or_generated = math.ceil(clients_needed_from_new_leads * leads_per_deal)
        monthly_leads_to_generate = math.ceil(total_leads_needed_or_generated / months_left_in_window)
        monthly_marketing_spend = monthly_leads_to_generate * actual_cpl
        needs_met = True # We have calculated a plan
    elif revenue_needed_from_new_leads_to_hit_goal <= 0: # Goal already met/exceeded by other sources
        total_leads_needed_or_generated = 0; monthly_leads_to_generate = 0; monthly_marketing_spend = 0
        needs_met = True # Goal met
    else: # Cannot meet goal in time/window
        total_leads_needed_or_generated = float('inf') if clients_needed_from_new_leads > 0 else 0
        monthly_leads_to_generate = float('inf') if clients_needed_from_new_leads > 0 else 0
        monthly_marketing_spend = float('inf') if clients_needed_from_new_leads > 0 else 0
        needs_met = False # Goal unreachable with this strategy

elif planning_strategy == "Set Marketing Budget":
    months_to_spend = max(1, 12 - start_gen_month_idx_num)
    monthly_marketing_spend = total_planned_marketing_spend / months_to_spend if months_to_spend > 0 else 0
    monthly_leads_to_generate = monthly_marketing_spend / actual_cpl if actual_cpl > 0 else 0
    total_leads_needed_or_generated = monthly_leads_to_generate * months_to_spend
    # end_gen_month_idx_num remains 11
    needs_met = True # Budget plan is always executable

# --- Build Monthly Projection Table ---
month_names = [(fy_start_dt_for_calc + pd.DateOffset(months=i)).strftime('%b %Y') for i in range(12)]
month_timestamps = [(fy_start_dt_for_calc + pd.DateOffset(months=i)) for i in range(12)]

projection_df = pd.DataFrame({
    'Month': month_names, 'Month_Timestamp': month_timestamps, 'Month Index': range(1, 13),
    'Revenue: Spillover (₹)': np.zeros(12, dtype=float), 'Revenue: Renewals (₹)': np.zeros(12, dtype=float),
    'Revenue: Pre-FY Leads (₹)': np.zeros(12, dtype=float), 'Revenue: New Leads This FY (₹)': np.zeros(12, dtype=float),
    'TOTAL REVENUE (Projected Monthly)': np.zeros(12, dtype=float), 'Clients Closing: Pre-FY Leads': np.zeros(12, dtype=float),
    'Clients Closing: New Leads This FY': np.zeros(12, dtype=float), 'Total New Clients Closing Monthly': np.zeros(12, dtype=float),
    'Leads Generated': np.zeros(12, dtype=float), 'Expense: Marketing (₹)': np.zeros(12, dtype=float),
    'Expense: Sales Salary (₹)': np.full(12, monthly_sales_salary_cost, dtype=float), 'Expense: Product Salary (₹)': np.full(12, monthly_product_salary_cost, dtype=float),
    'Expense: Services Salary (₹)': np.full(12, monthly_services_salary_cost, dtype=float), 'Expense: Server (₹)': np.full(12, monthly_server_cost, dtype=float),
    'TOTAL EXPENSES (Projected Monthly)': np.zeros(12, dtype=float), 'Monthly CAC Cost Base (₹)': np.zeros(12, dtype=float),
    'Monthly CAC (₹)': np.zeros(12, dtype=float), 'Monthly COGS (₹)': np.zeros(12, dtype=float),
    'Monthly Gross Profit (₹)': np.zeros(12, dtype=float), 'Monthly Gross Profit Margin (%)': np.zeros(12, dtype=float),
    'Monthly Product Exp % Revenue (%)': np.zeros(12, dtype=float), 'Monthly P/L (Projected)': np.zeros(12, dtype=float),
    'Cumulative P/L (Projected)': np.zeros(12, dtype=float)
})

# --- Populate DataFrame ---
# 1. Populate known/expected future revenue streams
may_idx = get_month_index("May", fy_start_month); jun_idx = get_month_index("Jun", fy_start_month); jul_idx = get_month_index("Jul", fy_start_month)
if 0 <= may_idx < 12: projection_df.loc[may_idx, 'Revenue: Spillover (₹)'] += spillover_may
if 0 <= jun_idx < 12: projection_df.loc[jun_idx, 'Revenue: Spillover (₹)'] += spillover_jun
if 0 <= jul_idx < 12: projection_df.loc[jul_idx, 'Revenue: Spillover (₹)'] += spillover_jul
if renewal_months_count > 0:
    for month_offset in range(renewal_months_count):
        idx = renewal_start_idx + month_offset
        if 0 <= idx < 12: projection_df.loc[idx, 'Revenue: Renewals (₹)'] += monthly_renewal_revenue
approx_gen_month_offset = -2 if fy_start_month=='Apr' else (1 if fy_start_month=='Jan' else (-5 if fy_start_month=='Jul' else -8))
closing_idx_prior = approx_gen_month_offset + avg_sales_cycle_months
if 0 <= closing_idx_prior < 12:
    projection_df.loc[closing_idx_prior, 'Revenue: Pre-FY Leads (₹)'] += revenue_from_prior_leads
    projection_df.loc[closing_idx_prior, 'Clients Closing: Pre-FY Leads'] += clients_from_prior_leads
elif closing_idx_prior >= 12 and leads_already_generated > 0: st.sidebar.warning(f"Pre-FY leads ({leads_already_generated}) might close *next* FY.")

# 2. Populate Leads/Spend based on Strategy & Calculate Resulting New Revenue
if needs_met and monthly_leads_to_generate != float('inf'):
    actual_end_gen_month_idx = int(min(end_gen_month_idx_num, 11))
    for gen_idx in range(start_gen_month_idx_num, actual_end_gen_month_idx + 1):
        if gen_idx < 0: continue
        projection_df.loc[gen_idx, 'Leads Generated'] = monthly_leads_to_generate
        projection_df.loc[gen_idx, 'Expense: Marketing (₹)'] = monthly_marketing_spend
        closing_idx = gen_idx + avg_sales_cycle_months
        if closing_idx < 12:
            clients_batch = projection_df.loc[gen_idx, 'Leads Generated'] * conversion_rate_decimal
            revenue_batch = clients_batch * avg_ticket_size
            projection_df.loc[closing_idx, 'Revenue: New Leads This FY (₹)'] += revenue_batch
            projection_df.loc[closing_idx, 'Clients Closing: New Leads This FY'] += clients_batch

# 3. Calculate Intermediate & Total Columns for Projected Months
projection_df['Total New Clients Closing Monthly'] = projection_df['Clients Closing: Pre-FY Leads'] + projection_df['Clients Closing: New Leads This FY']
projection_df['TOTAL REVENUE (Projected Monthly)'] = projection_df[['Revenue: Spillover (₹)', 'Revenue: Renewals (₹)', 'Revenue: Pre-FY Leads (₹)', 'Revenue: New Leads This FY (₹)']].sum(axis=1)
projection_df['TOTAL EXPENSES (Projected Monthly)'] = projection_df[['Expense: Marketing (₹)', 'Expense: Sales Salary (₹)', 'Expense: Product Salary (₹)', 'Expense: Services Salary (₹)', 'Expense: Server (₹)']].sum(axis=1)
projection_df['Monthly CAC Cost Base (₹)'] = projection_df['Expense: Marketing (₹)'] + projection_df['Expense: Sales Salary (₹)']
projection_df['Monthly CAC (₹)'] = projection_df['Monthly CAC Cost Base (₹)'].div(projection_df['Total New Clients Closing Monthly']).replace([np.inf, -np.inf], 0).fillna(0)
projection_df['Monthly COGS (₹)'] = projection_df['Expense: Services Salary (₹)'] + projection_df['Expense: Server (₹)']
projection_df['Monthly Gross Profit (₹)'] = projection_df['TOTAL REVENUE (Projected Monthly)'] - projection_df['Monthly COGS (₹)']
projection_df['Monthly Gross Profit Margin (%)'] = projection_df['Monthly Gross Profit (₹)'].div(projection_df['TOTAL REVENUE (Projected Monthly)']).replace([np.inf, -np.inf], 0).fillna(0) * 100
projection_df['Monthly Product Exp % Revenue (%)'] = projection_df['Expense: Product Salary (₹)'].div(projection_df['TOTAL REVENUE (Projected Monthly)']).replace([np.inf, -np.inf], 0).fillna(0) * 100
projection_df['Monthly P/L (Projected)'] = projection_df['TOTAL REVENUE (Projected Monthly)'] - projection_df['TOTAL EXPENSES (Projected Monthly)']
projection_df['Cumulative P/L (Projected)'] = projection_df['Monthly P/L (Projected)'].cumsum()

# --- Calculate Final FY KPIs & Metrics ---
projected_revenue_full_fy = projection_df['TOTAL REVENUE (Projected Monthly)'].sum()
total_fy_revenue_incl_actuals = projected_revenue_full_fy + actual_revenue_booked # Final total
total_fy_expenses = projection_df['TOTAL EXPENSES (Projected Monthly)'].sum()
total_fy_pl = total_fy_revenue_incl_actuals - total_fy_expenses # Final P/L based on total rev
target_delta_display = total_fy_revenue_incl_actuals - total_revenue_target # Final comparison vs Goal

# Other FY metrics calculation remain the same, using updated totals where needed
total_fy_new_clients = projection_df['Total New Clients Closing Monthly'].sum()
total_fy_mktg_spend = projection_df['Expense: Marketing (₹)'].sum()
total_fy_sales_salary = projection_df['Expense: Sales Salary (₹)'].sum()
total_fy_product_salary = projection_df['Expense: Product Salary (₹)'].sum()
total_fy_cogs = projection_df['Monthly COGS (₹)'].sum()
total_fy_gross_profit = total_fy_revenue_incl_actuals - total_fy_cogs # Use final total revenue
overall_fy_cac = (total_fy_mktg_spend + total_fy_sales_salary) / total_fy_new_clients if total_fy_new_clients > 0 else 0
overall_fy_gross_profit_margin = (total_fy_gross_profit / total_fy_revenue_incl_actuals) * 100 if total_fy_revenue_incl_actuals > 0 else 0
overall_fy_product_exp_perc = (total_fy_product_salary / total_fy_revenue_incl_actuals) * 100 if total_fy_revenue_incl_actuals > 0 else 0
overall_fy_rev_per_employee = total_fy_revenue_incl_actuals / total_headcount if total_headcount > 0 else 0
ltv_cac_ratio = ltv_per_customer / overall_fy_cac if overall_fy_cac > 0 else 0

# --- Display KPIs ---
st.header("📊 Key Performance Indicators (KPIs)")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
kpi_col1.metric("Overall Revenue Target (Goal)", f"₹{total_revenue_target/100000:.1f} L")
# Use the simplified definition for Target Remaining KPI
kpi_col2.metric("Target Remaining (Goal - Actual Booked)", f"₹{target_remaining_kpi/100000:.1f} L", help="Overall Goal Target - Actual Revenue Booked So Far")
kpi_col3.metric("Leads Needed Per Deal", f"{leads_per_deal:.0f}", help=f"Based on {lead_to_deal_conversion_rate_percent}% conversion")
kpi_col4.metric("Adjusted CPL", f"₹{actual_cpl:,.0f}", help=f"{cpl_increase_percent:.1f}% increase on base ₹{base_cpl_rupees:,.0f}")

st.markdown("---")

# --- Conditional Display: Lead Gen Requirements or Marketing Plan ---
if planning_strategy == "Set Revenue Target":
    st.header("🎯 Lead Generation Required (To Hit Overall Goal)")
    # Use the correct gap variable calculated earlier
    revenue_gap_display = revenue_needed_from_new_leads_to_hit_goal
    clients_needed_display = math.ceil(revenue_gap_display / avg_ticket_size if avg_ticket_size > 0 else 0)

    if revenue_gap_display <= 0 : # Check if the gap *for new leads* is zero or less
        st.success(f"✅ Goal Likely Met/Exceeded: Based on Actuals + Expected Future Revenue (Spillover, Renewals, Pre-FY Leads), no *additional* revenue is required from *new* leads this FY to hit the goal (₹{total_revenue_target/100000:.1f} L).")
    elif not needs_met: # Needs_met was set based on the gap calculation
        st.error(f"⚠️ **Overall Goal Likely Unreachable Within FY**")
        st.markdown(f"Remaining revenue needed specifically from *new leads*: **₹{revenue_gap_display/100000:.2f} L** ({clients_needed_display} clients)")
        # ... (Reasons remain the same) ...
        if last_gen_month_index_for_fy <= 0 and avg_sales_cycle_months >= 12: st.markdown(f"Reason: {avg_sales_cycle_months}-mo. cycle.")
        elif current_month_index > last_gen_month_index_for_fy :
            last_gen_month_str = (fy_start_dt_for_calc + pd.DateOffset(months=end_gen_month_idx_num)).strftime('%b %Y') if end_gen_month_idx_num >=0 else "N/A"
            st.markdown(f"Reason: Currently Month {current_month_index}. Leads needed by {last_gen_month_str} for {avg_sales_cycle_months}-mo. cycle.")
        elif leads_per_deal == float('inf'): st.markdown("Reason: 0% conversion.")
        else: st.markdown("Reason: Insufficient time or required lead volume too high.")
    else: # Needs_met is True and new leads *are* required
        req_col_a, req_col_b, req_col_c = st.columns(3)
        st.info(f"To generate the required **₹{revenue_gap_display/100000:.2f} L** ({clients_needed_display} clients) from *new leads* by end of FY:")
        last_gen_month_str = (fy_start_dt_for_calc + pd.DateOffset(months=end_gen_month_idx_num)).strftime('%b %Y') if end_gen_month_idx_num >=0 else "N/A"
        req_col_a.metric("Lead Gen Window Ends", f"Month {last_gen_month_index_for_fy} ({last_gen_month_str})")
        req_col_b.metric(f"Avg. Monthly Leads Needed (Now - {last_gen_month_str})", f"{monthly_leads_to_generate:.0f}", help=f"Total {total_leads_needed_or_generated:.0f} leads needed over {months_left_in_window} months")
        req_col_c.metric("Avg. Monthly Marketing Spend (During Window)", f"₹{monthly_marketing_spend:,.0f}")

elif planning_strategy == "Set Marketing Budget":
    st.header("📈 Marketing Plan & Revenue Projection")
    plan_col_a, plan_col_b, plan_col_c = st.columns(3)
    plan_col_a.metric("Total Planned Marketing Spend", f"₹{total_planned_marketing_spend/100000:.1f} L")
    plan_col_b.metric("Avg. Monthly Marketing Spend (Planned)", f"₹{monthly_marketing_spend:,.0f}", help=f"Spread over {max(1, 12 - start_gen_month_idx_num)} remaining months")
    plan_col_c.metric("Avg. Monthly Leads Generated (Planned)", f"{monthly_leads_to_generate:.1f}")

# --- Display Final FY Summary & SaaS Metrics ---
st.markdown("---")
st.header("🏁 Projected Financial Year Summary (Incl. Actuals Booked)")
final_kpi1, final_kpi2, final_kpi3 = st.columns(3)
final_kpi1.metric("Projected Total Revenue (Incl. Actuals)", f"₹{total_fy_revenue_incl_actuals/100000:.1f} L", f"₹{target_delta_display/100000:.1f} L vs Goal") # Compares final projection to goal
final_kpi2.metric("Projected Total Expenses", f"₹{total_fy_expenses/100000:.1f} L")
final_kpi3.metric("Projected Profit/Loss", f"₹{total_fy_pl/100000:.1f} L")

st.subheader("Key SaaS Metrics (Projected for Full FY)")
metric_col1, metric_col2, metric_col3 = st.columns(3); metric_col4, metric_col5, metric_col6 = st.columns(3)
metric_col1.metric("LTV", f"₹{ltv_per_customer/100000:.1f} L", help="(AMC/Yr*Yrs)+Phase2")
metric_col2.metric("CAC", f"₹{overall_fy_cac:,.0f}", help="(Mktg+SalesSal)/NewClients")
metric_col3.metric("LTV:CAC Ratio", f"{ltv_cac_ratio:.1f}:1")
metric_col4.metric("Gross Profit Margin", f"{overall_fy_gross_profit_margin:.1f}%", help="(Rev-COGS)/Rev")
metric_col5.metric("Product Exp % Rev", f"{overall_fy_product_exp_perc:.1f}%", help="(ProdSal/Rev)*100")
metric_col6.metric("Rev/Employee", f"₹{overall_fy_rev_per_employee/100000:.1f} L", help="TotalRev/Headcount")

# --- Display Monthly Projection Table ---
st.header(f"🗓️ Monthly Financial Projection (FY: {fy_start_month} 2025 - {fy_end_str})")
display_table_df = projection_df.copy().reset_index(drop=True)
cols_to_display = ['Month', 'TOTAL REVENUE (Projected Monthly)', 'Total New Clients Closing Monthly', 'Monthly CAC (₹)', 'Monthly Gross Profit Margin (%)', 'TOTAL EXPENSES (Projected Monthly)', 'Monthly P/L (Projected)', 'Cumulative P/L (Projected)', 'Leads Generated']
display_table_df = display_table_df[cols_to_display]
# Apply formatting... (same as before)
display_table_df['TOTAL REVENUE (Projected Monthly)'] = display_table_df['TOTAL REVENUE (Projected Monthly)'].apply(lambda x: f"₹{x:,.0f}")
display_table_df['Total New Clients Closing Monthly'] = display_table_df['Total New Clients Closing Monthly'].apply(lambda x: f"{x:.1f}")
display_table_df['Monthly CAC (₹)'] = display_table_df['Monthly CAC (₹)'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) and x > 0 and x != float('inf') else "N/A")
display_table_df['Monthly Gross Profit Margin (%)'] = display_table_df['Monthly Gross Profit Margin (%)'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) and x != float('inf') else "N/A")
display_table_df['TOTAL EXPENSES (Projected Monthly)'] = display_table_df['TOTAL EXPENSES (Projected Monthly)'].apply(lambda x: f"₹{x:,.0f}")
display_table_df['Monthly P/L (Projected)'] = display_table_df['Monthly P/L (Projected)'].apply(lambda x: f"₹{x:,.0f}")
display_table_df['Cumulative P/L (Projected)'] = display_table_df['Cumulative P/L (Projected)'].apply(lambda x: f"₹{x:,.0f}")
display_table_df['Leads Generated'] = display_table_df['Leads Generated'].apply(lambda x: f"{x:.0f}" if pd.notna(x) and x != float('inf') else "N/A")
st.dataframe(display_table_df.set_index('Month'), use_container_width=True)

# --- Charts ---
st.markdown("---"); st.header("📊 Charts"); st.caption("Charts use the underlying datetime for x-axis ordering.")
plot_df = projection_df.set_index('Month_Timestamp')
st.subheader("Revenue Breakdown (Projected Monthly)"); revenue_cols = ['Revenue: Spillover (₹)', 'Revenue: Renewals (₹)', 'Revenue: Pre-FY Leads (₹)', 'Revenue: New Leads This FY (₹)']; st.bar_chart(plot_df[revenue_cols])
st.subheader("Profitability (Projected Monthly)"); st.line_chart(plot_df[['Cumulative P/L (Projected)', 'Monthly P/L (Projected)']])
st.subheader("Expense Breakdown (Projected Monthly)"); expense_cols = ['Expense: Marketing (₹)', 'Expense: Sales Salary (₹)', 'Expense: Product Salary (₹)', 'Expense: Services Salary (₹)', 'Expense: Server (₹)']; st.bar_chart(plot_df[expense_cols])
st.subheader("Customer Acquisition Cost (Monthly)"); cac_chart_data = plot_df[['Monthly CAC (₹)']].copy(); cac_chart_data = cac_chart_data[cac_chart_data['Monthly CAC (₹)'] > 0]
if not cac_chart_data.empty: st.line_chart(cac_chart_data)
else: st.caption("No new clients projected to close in months with marketing/sales costs.")

# --- Notes ---
st.markdown("---"); st.subheader("Assumptions & Notes:")
# (Notes generation remains the same)
prior_lead_closing_month_str = 'N/A'; start_gen_month_str = 'N/A'; end_gen_month_str = 'N/A'; renewal_start_str = 'N/A'; renewal_end_str = 'N/A'; may_month_str_note = 'N/A'; jun_month_str_note = 'N/A'; jul_month_str_note = 'N/A'; current_month_str_note = 'N/A'
try:
    if 0 <= closing_idx_prior < 12: prior_lead_closing_month_str = f"around {(fy_start_dt_for_calc + pd.DateOffset(months=closing_idx_prior)).strftime('%b %Y')}"
    elif closing_idx_prior >= 12: prior_lead_closing_month_str = "next FY (projected)"
    elif closing_idx_prior < 0: prior_lead_closing_month_str = "last FY (projected)"
    start_gen_month_str = (fy_start_dt_for_calc + pd.DateOffset(months=start_gen_month_idx_num)).strftime('%b %Y') if start_gen_month_idx_num >=0 else 'N/A'
    if planning_strategy == "Set Revenue Target": end_gen_month_str = (fy_start_dt_for_calc + pd.DateOffset(months=end_gen_month_idx_num)).strftime('%b %Y') if end_gen_month_idx_num >=0 and last_gen_month_index_for_fy > 0 else 'N/A'
    else: end_gen_month_str = (fy_start_dt_for_calc + pd.DateOffset(months=11)).strftime('%b %Y')
    renewal_start_str = (fy_start_dt_for_calc + pd.DateOffset(months=renewal_start_idx)).strftime('%b %Y') if renewal_start_idx >=0 else 'N/A'
    renewal_end_str = (fy_start_dt_for_calc + pd.DateOffset(months=renewal_end_idx)).strftime('%b %Y') if renewal_end_idx >=0 else 'N/A'
    may_month_str_note = (fy_start_dt_for_calc + pd.DateOffset(months=get_month_index("May", fy_start_month))).strftime('%b %Y')
    jun_month_str_note = (fy_start_dt_for_calc + pd.DateOffset(months=get_month_index("Jun", fy_start_month))).strftime('%b %Y')
    jul_month_str_note = (fy_start_dt_for_calc + pd.DateOffset(months=get_month_index("Jul", fy_start_month))).strftime('%b %Y')
    current_month_str_note = (fy_start_dt_for_calc + pd.DateOffset(months=current_month_index-1)).strftime('%b %Y')
except Exception: pass
st.markdown(f"""
*   **Strategy:** Using **{planning_strategy}**. The Overall Target is the goal; calculations project towards/from it.
*   Financial Year: **{fy_start_month} 2025** to **{fy_end_str}**. Current Month: **{current_month_str_note}**.
*   New Business: Ticket **₹{avg_ticket_size/100000:.1f}L**, Cycle **{avg_sales_cycle_months} mo.**, Conversion **{lead_to_deal_conversion_rate_percent}%**.
*   Future Revenue: Spillover ({may_month_str_note}-{jul_month_str_note}), Renewals ({renewal_start_str}-{renewal_end_str}), Pre-FY Leads ({prior_lead_closing_month_str}).
*   Lead Gen/Spend: From **{start_gen_month_str}** to **{end_gen_month_str}** based on strategy.
*   **Actual Revenue Booked** (₹{actual_revenue_booked/100000:.1f}L) reduces the gap to the Overall Goal.
*   **LTV**=**₹{ltv_per_customer/100000:.1f} L**. **CAC**(FY Avg):**₹{overall_fy_cac:,.0f}**. **LTV:CAC**:**{ltv_cac_ratio:.1f}:1**.
*   **Gross Profit Margin**(FY Avg):**{overall_fy_gross_profit_margin:.1f}%**. **Product Exp%Rev**(FY Avg):**{overall_fy_product_exp_perc:.1f}%**. **Rev/Emp**(FY Avg):**₹{overall_fy_rev_per_employee/100000:.1f} L**.
*   Projections are estimates.
""")