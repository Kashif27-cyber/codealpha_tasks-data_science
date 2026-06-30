"""
╔══════════════════════════════════════════════════════════════════════╗
║      UNEMPLOYMENT ANALYSIS IN INDIA — COMPLETE PYTHON PROJECT        ║
║      EDA · Covid Impact · State Trends · Policy Insights             ║
╚══════════════════════════════════════════════════════════════════════╝

Dataset : Kaggle — gokulrajkmv/unemployment-in-india
Author  : <Your Name>
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
PLOTS = '../plots'
os.makedirs(PLOTS, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────
PRE   = '#2196F3'   # blue   – pre-covid
COVID = '#F44336'   # red    – covid peak
POST  = '#4CAF50'   # green  – recovery
GOLD  = '#FF9800'
BG    = '#F8F9FF'

def save(name):
    plt.savefig(f'{PLOTS}/{name}', dpi=150, bbox_inches='tight',
                facecolor=BG)
    plt.close()
    print(f'  ✅ {name}')

print("="*65)
print("   UNEMPLOYMENT ANALYSIS IN INDIA — PYTHON PROJECT")
print("="*65)

# ════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN DATA
# ════════════════════════════════════════════════════════════════
print("\n── 1. LOAD & CLEAN ──")
df = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')
df.columns = df.columns.str.strip()

# Parse date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month
df['Year']  = df['Date'].dt.year
df['YearMonth'] = df['Date'].dt.to_period('M')
df['MonthName'] = df['Date'].dt.strftime('%b')

# Covid flag
df['Covid_Period'] = df['Date'].apply(
    lambda d: 'Lockdown' if (d.year==2020 and d.month in [4,5])
              else ('Recovery' if (d.year==2020 and d.month>=6)
              else 'Pre-Covid'))

col_rate = 'Estimated Unemployment Rate (%)'
col_lpr  = 'Estimated Labour Participation Rate (%)'
col_emp  = 'Estimated Employed'

print(f"  Shape          : {df.shape}")
print(f"  Date range     : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  States         : {df['Region'].nunique()}")
print(f"  Missing values : {df.isnull().sum().sum()}")
print(f"\n{df[[col_rate, col_lpr]].describe().round(2)}")


# ════════════════════════════════════════════════════════════════
# 2. NATIONAL TREND (Plot 1)
# ════════════════════════════════════════════════════════════════
print("\n── 2. NATIONAL TREND ──")
national = df.groupby('Date')[col_rate].mean().reset_index()

fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
ax.set_facecolor(BG)

# Shade regions
ax.axvspan(pd.Timestamp('2020-04-01'), pd.Timestamp('2020-05-31'),
           color=COVID, alpha=0.15, label='Lockdown (Apr–May 2020)')
ax.axvspan(pd.Timestamp('2020-06-01'), national['Date'].max(),
           color=POST, alpha=0.10, label='Recovery Phase')

ax.plot(national['Date'], national[col_rate], color=PRE,
        lw=2.5, marker='o', markersize=5, label='Avg Unemployment Rate')
ax.fill_between(national['Date'], national[col_rate],
                alpha=0.2, color=PRE)

# Annotations
peak = national.loc[national[col_rate].idxmax()]
ax.annotate(f'PEAK\n{peak[col_rate]:.1f}%',
            xy=(peak['Date'], peak[col_rate]),
            xytext=(peak['Date'], peak[col_rate]+3),
            arrowprops=dict(arrowstyle='->', color=COVID, lw=1.5),
            fontsize=9, color=COVID, fontweight='bold', ha='center')

ax.set_title('India National Unemployment Rate — Jan 2019 to Nov 2020',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Date'); ax.set_ylabel('Unemployment Rate (%)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); save('01_national_trend.png')


# ════════════════════════════════════════════════════════════════
# 3. PRE vs COVID COMPARISON (Plot 2)
# ════════════════════════════════════════════════════════════════
print("── 3. PRE vs COVID COMPARISON ──")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
fig.suptitle('Pre-Covid vs Lockdown vs Recovery Comparison',
             fontsize=14, fontweight='bold')

period_stats = df.groupby('Covid_Period')[col_rate].agg(['mean','std','median'])
order = ['Pre-Covid', 'Lockdown', 'Recovery']
colors_p = [PRE, COVID, POST]

# Bar chart – mean rate
ax = axes[0]; ax.set_facecolor(BG)
bars = ax.bar(order,
              [period_stats.loc[p,'mean'] for p in order],
              color=colors_p, edgecolor='white', width=0.5)
for bar, v in zip(bars, [period_stats.loc[p,'mean'] for p in order]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax.set_title('Mean Unemployment Rate'); ax.set_ylabel('%'); ax.grid(axis='y', alpha=0.3)

# Violin plot
ax = axes[1]; ax.set_facecolor(BG)
data_vio = [df[df['Covid_Period']==p][col_rate].values for p in order]
vp = ax.violinplot(data_vio, positions=[1,2,3], showmedians=True)
for i, (body, c) in enumerate(zip(vp['bodies'], colors_p)):
    body.set_facecolor(c); body.set_alpha(0.7)
ax.set_xticks([1,2,3]); ax.set_xticklabels(order, fontsize=9)
ax.set_title('Distribution of Unemployment Rates')
ax.set_ylabel('%'); ax.grid(axis='y', alpha=0.3)

# Box plot — LPR
ax = axes[2]; ax.set_facecolor(BG)
data_lpr = [df[df['Covid_Period']==p][col_lpr].values for p in order]
bp = ax.boxplot(data_lpr, labels=order, patch_artist=True,
                medianprops=dict(color='black', lw=2))
for patch, c in zip(bp['boxes'], colors_p):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_title('Labour Participation Rate'); ax.set_ylabel('%')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout(); save('02_pre_vs_covid_comparison.png')


# ════════════════════════════════════════════════════════════════
# 4. STATE-WISE ANALYSIS (Plot 3)
# ════════════════════════════════════════════════════════════════
print("── 4. STATE-WISE HEATMAP ──")
state_monthly = df.pivot_table(
    index='Region', columns='YearMonth',
    values=col_rate, aggfunc='mean')

fig, ax = plt.subplots(figsize=(20, 12), facecolor=BG)
sns.heatmap(state_monthly, cmap='RdYlGn_r', ax=ax,
            linewidths=0.3, linecolor='#e0e0e0',
            cbar_kws={'label': 'Unemployment Rate (%)'},
            annot=False)
ax.set_title('State-wise Unemployment Rate Heatmap (Jan 2019 – Nov 2020)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Month-Year', fontsize=10)
ax.set_ylabel('State', fontsize=10)
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', labelsize=8)
# Mark covid boundary
cols = list(state_monthly.columns)
covid_col = next((i for i,c in enumerate(cols) if str(c)=='2020-04'), None)
if covid_col:
    ax.axvline(covid_col, color=COVID, lw=2.5, linestyle='--')
    ax.text(covid_col+0.2, -1.5, '← Covid →', color=COVID,
            fontsize=9, fontweight='bold')
plt.tight_layout(); save('03_state_heatmap.png')


# ════════════════════════════════════════════════════════════════
# 5. TOP/BOTTOM STATES (Plot 4)
# ════════════════════════════════════════════════════════════════
print("── 5. TOP & BOTTOM STATES ──")
pre  = df[df['Covid_Period']=='Pre-Covid'].groupby('Region')[col_rate].mean()
lock = df[df['Covid_Period']=='Lockdown'].groupby('Region')[col_rate].mean()
impact = (lock - pre).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)

# Left: avg unemployment ranking
avg_state = df.groupby('Region')[col_rate].mean().sort_values(ascending=True)
colors_bar = [COVID if v > avg_state.median()*1.5
              else (PRE if v < avg_state.median()*0.8 else GOLD)
              for v in avg_state]
axes[0].set_facecolor(BG)
axes[0].barh(avg_state.index, avg_state.values, color=colors_bar, edgecolor='white')
axes[0].axvline(avg_state.median(), color='black', lw=1.5, linestyle='--',
                label=f'Median: {avg_state.median():.1f}%')
axes[0].set_title('Average Unemployment Rate by State\n(Jan 2019 – Nov 2020)',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Avg Unemployment Rate (%)'); axes[0].legend()
axes[0].grid(axis='x', alpha=0.3)

# Right: Covid impact (% increase)
axes[1].set_facecolor(BG)
impact_sorted = impact.sort_values(ascending=True)
bar_colors = [COVID if v > impact.mean() else POST for v in impact_sorted]
axes[1].barh(impact_sorted.index, impact_sorted.values,
             color=bar_colors, edgecolor='white')
axes[1].axvline(0, color='black', lw=1)
axes[1].set_title('Covid-19 Impact: Unemployment Increase\n(Pre-Covid → Lockdown)',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Increase in Unemployment Rate (pp)')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout(); save('04_state_ranking_covid_impact.png')


# ════════════════════════════════════════════════════════════════
# 6. SEASONAL TRENDS (Plot 5)
# ════════════════════════════════════════════════════════════════
print("── 6. SEASONAL TRENDS ──")
pre_df = df[df['Covid_Period']=='Pre-Covid']
monthly_avg = pre_df.groupby('Month')[col_rate].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor=BG)
fig.suptitle('Seasonal Unemployment Patterns (Pre-Covid 2019)',
             fontsize=13, fontweight='bold')

ax = axes[0]; ax.set_facecolor(BG)
bar_c = [COVID if v==monthly_avg.max() else (POST if v==monthly_avg.min() else PRE)
         for v in monthly_avg]
ax.bar([month_names[m-1] for m in monthly_avg.index], monthly_avg.values,
       color=bar_c, edgecolor='white')
ax.set_title('Monthly Average Unemployment Rate (Pre-Covid)')
ax.set_ylabel('%'); ax.grid(axis='y', alpha=0.3)
ax.annotate(f'Peak: {monthly_avg.max():.1f}%',
            xy=(monthly_avg.idxmax()-1, monthly_avg.max()),
            xytext=(monthly_avg.idxmax()+0.5, monthly_avg.max()+0.5),
            arrowprops=dict(arrowstyle='->', color=COVID),
            color=COVID, fontweight='bold', fontsize=9)

# Polar / radar chart approximation using fill
ax = axes[1]; ax.set_facecolor(BG)
vals = list(monthly_avg.values) + [monthly_avg.values[0]]  # close loop
months_full = [month_names[m-1] for m in monthly_avg.index] + [month_names[0]]
ax.plot(months_full, vals[:-1]+[vals[-1]], color=PRE, lw=2, marker='o')
ax.fill_between(months_full, vals[:-1]+[vals[-1]], alpha=0.2, color=PRE)
ax.set_title('Seasonal Pattern Across Months')
ax.set_ylabel('Unemployment Rate (%)'); ax.grid(alpha=0.3)

plt.tight_layout(); save('05_seasonal_trends.png')


# ════════════════════════════════════════════════════════════════
# 7. URBAN vs RURAL (Plot 6)
# ════════════════════════════════════════════════════════════════
print("── 7. URBAN vs RURAL ──")
if 'Area' in df.columns:
    area_time = df.groupby(['Date','Area'])[col_rate].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor=BG)
    fig.suptitle('Urban vs Rural Unemployment Analysis', fontsize=13, fontweight='bold')

    for area, color in [('Urban','#E91E63'), ('Rural','#009688')]:
        sub = area_time[area_time['Area']==area]
        axes[0].plot(sub['Date'], sub[col_rate], label=area,
                     color=color, lw=2, marker='o', markersize=4)
    axes[0].set_facecolor(BG)
    axes[0].axvspan(pd.Timestamp('2020-04-01'), pd.Timestamp('2020-05-31'),
                    color=COVID, alpha=0.12)
    axes[0].set_title('Unemployment Rate Over Time by Area')
    axes[0].set_ylabel('%'); axes[0].legend(); axes[0].grid(alpha=0.3)

    area_period = df.groupby(['Covid_Period','Area'])[col_rate].mean().reset_index()
    area_pivot = area_period.pivot(index='Covid_Period', columns='Area', values=col_rate)
    area_pivot.loc[order].plot(kind='bar', ax=axes[1], color=['#E91E63','#009688'],
                               edgecolor='white', width=0.6)
    axes[1].set_facecolor(BG)
    axes[1].set_title('Mean Rate by Period and Area Type')
    axes[1].set_ylabel('%'); axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(title='Area'); axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout(); save('06_urban_vs_rural.png')


# ════════════════════════════════════════════════════════════════
# 8. LABOUR PARTICIPATION vs UNEMPLOYMENT (Plot 7)
# ════════════════════════════════════════════════════════════════
print("── 8. LPR vs UNEMPLOYMENT RATE ──")
fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor=BG)
fig.suptitle('Labour Participation Rate vs Unemployment Rate', fontsize=13, fontweight='bold')

colors_map = {'Pre-Covid': PRE, 'Lockdown': COVID, 'Recovery': POST}
for period, grp in df.groupby('Covid_Period'):
    axes[0].scatter(grp[col_lpr], grp[col_rate],
                    color=colors_map[period], alpha=0.4, s=20, label=period)
axes[0].set_facecolor(BG)
axes[0].set_xlabel('Labour Participation Rate (%)')
axes[0].set_ylabel('Unemployment Rate (%)')
axes[0].set_title('LPR vs Unemployment (coloured by period)')
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Correlation over time
corr_monthly = df.groupby('Date').apply(
    lambda g: g[[col_rate, col_lpr]].corr().iloc[0,1]).reset_index()
corr_monthly.columns = ['Date', 'Correlation']
axes[1].bar(corr_monthly['Date'], corr_monthly['Correlation'],
            width=20, color=[COVID if v<0 else PRE for v in corr_monthly['Correlation']],
            edgecolor='none')
axes[1].set_facecolor(BG)
axes[1].axhline(0, color='black', lw=1)
axes[1].set_title('Monthly Correlation: Unemployment vs LPR')
axes[1].set_ylabel('Pearson Correlation'); axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout(); save('07_lpr_vs_unemployment.png')


# ════════════════════════════════════════════════════════════════
# 9. RECOVERY ANALYSIS (Plot 8)
# ════════════════════════════════════════════════════════════════
print("── 9. RECOVERY TRAJECTORY ──")
national_all = df.groupby('Date')[col_rate].mean().reset_index()
pre_mean = national_all[national_all['Date'] < '2020-04-01'][col_rate].mean()
peak_val = national_all[col_rate].max()

fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
ax.set_facecolor(BG)

ax.plot(national_all['Date'], national_all[col_rate], color='#424242',
        lw=2.5, zorder=5, label='National Avg')
ax.fill_between(national_all['Date'], national_all[col_rate],
                where=(national_all['Date'] >= '2020-04-01') &
                      (national_all['Date'] <= '2020-05-31'),
                color=COVID, alpha=0.5, label='Lockdown Spike')
ax.fill_between(national_all['Date'], national_all[col_rate],
                where=national_all['Date'] > '2020-05-31',
                color=POST, alpha=0.3, label='Recovery')
ax.axhline(pre_mean, color=PRE, lw=1.5, linestyle='--',
           label=f'Pre-Covid Avg: {pre_mean:.1f}%')
ax.axhline(peak_val, color=COVID, lw=1.5, linestyle=':',
           label=f'Peak: {peak_val:.1f}%')

# Recovery arrow
rec_start = national_all[national_all['Date']=='2020-06-30']
if not rec_start.empty:
    rec_val = rec_start[col_rate].values[0]
    ax.annotate('', xy=(national_all['Date'].iloc[-1], national_all[col_rate].iloc[-1]),
                xytext=(rec_start['Date'].values[0], rec_val),
                arrowprops=dict(arrowstyle='->', color=POST, lw=2))

ax.set_title('Unemployment Recovery Trajectory Post-Lockdown',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Unemployment Rate (%)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); save('08_recovery_trajectory.png')


# ════════════════════════════════════════════════════════════════
# 10. CORRELATION HEATMAP (Plot 9)
# ════════════════════════════════════════════════════════════════
print("── 10. CORRELATION HEATMAP ──")
num_df = df[['Month', 'Year', col_rate, col_lpr, col_emp]].copy()
num_df['Covid_Flag'] = (df['Covid_Period'] != 'Pre-Covid').astype(int)
num_df['Lockdown_Flag'] = (df['Covid_Period'] == 'Lockdown').astype(int)

fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, linewidths=0.5, square=True,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout(); save('09_correlation_heatmap.png')


# ════════════════════════════════════════════════════════════════
# 11. POLICY INSIGHTS DASHBOARD (Plot 10)
# ════════════════════════════════════════════════════════════════
print("── 11. POLICY INSIGHTS DASHBOARD ──")
fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle('Unemployment Analysis — Policy Insights Dashboard',
             fontsize=16, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 11a: Top 5 most impacted states
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(BG)
top5 = impact.nlargest(5)
bars = ax1.bar(top5.index, top5.values, color=COVID, edgecolor='white')
ax1.set_title('Top 5 States\nMost Hit by Covid', fontweight='bold', fontsize=10)
ax1.set_ylabel('Unemployment Increase (pp)')
ax1.tick_params(axis='x', rotation=30, labelsize=7)
ax1.grid(axis='y', alpha=0.3)

# 11b: Fastest recovery states
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(BG)
rec_df = df[df['Covid_Period']=='Recovery']
lock_df = df[df['Covid_Period']=='Lockdown']
rec_rate = rec_df.groupby('Region')[col_rate].mean()
lock_rate = lock_df.groupby('Region')[col_rate].mean()
recovery_speed = (lock_rate - rec_rate).dropna().sort_values(ascending=False).head(5)
ax2.bar(recovery_speed.index, recovery_speed.values, color=POST, edgecolor='white')
ax2.set_title('Top 5 States\nFastest Recovery', fontweight='bold', fontsize=10)
ax2.set_ylabel('Rate Decline (pp)')
ax2.tick_params(axis='x', rotation=30, labelsize=7)
ax2.grid(axis='y', alpha=0.3)

# 11c: Monthly average all years
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor(BG)
for yr, grp in df.groupby('Year'):
    mon_avg = grp.groupby('Month')[col_rate].mean()
    ax3.plot(mon_avg.index, mon_avg.values,
             marker='o', label=str(yr), lw=2)
ax3.set_xticks(range(1,13))
ax3.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=8)
ax3.set_title('Month-wise Rate by Year', fontweight='bold', fontsize=10)
ax3.set_ylabel('%'); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

# 11d: Distribution pre vs lockdown
ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor(BG)
pre_rates  = df[df['Covid_Period']=='Pre-Covid'][col_rate]
lock_rates = df[df['Covid_Period']=='Lockdown'][col_rate]
ax4.hist(pre_rates,  bins=25, alpha=0.6, color=PRE,   label=f'Pre-Covid (μ={pre_rates.mean():.1f}%)')
ax4.hist(lock_rates, bins=25, alpha=0.6, color=COVID,  label=f'Lockdown (μ={lock_rates.mean():.1f}%)')
ax4.set_title('Rate Distribution Shift', fontweight='bold', fontsize=10)
ax4.set_xlabel('%'); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

# 11e: Linear trend (regression)
ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor(BG)
nat = df.groupby('Date')[col_rate].mean().reset_index()
nat['Day_Num'] = (nat['Date'] - nat['Date'].min()).dt.days
X = nat[['Day_Num']]
y = nat[col_rate]
lr = LinearRegression().fit(X, y)
ax5.scatter(nat['Date'], nat[col_rate], s=20, alpha=0.6, color=PRE)
ax5.plot(nat['Date'], lr.predict(X), color=COVID, lw=2, linestyle='--', label='Trend')
ax5.set_title('Overall Unemployment Trend\n(Linear Regression)', fontweight='bold', fontsize=10)
ax5.set_ylabel('%'); ax5.legend(); ax5.grid(alpha=0.3)

# 11f: Key metrics summary
ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor(BG)
ax6.axis('off')
metrics = [
    ('Pre-Covid Avg Rate',    f'{df[df["Covid_Period"]=="Pre-Covid"][col_rate].mean():.1f}%'),
    ('Lockdown Avg Rate',     f'{df[df["Covid_Period"]=="Lockdown"][col_rate].mean():.1f}%'),
    ('Recovery Avg Rate',     f'{df[df["Covid_Period"]=="Recovery"][col_rate].mean():.1f}%'),
    ('Peak Unemployment',     f'{df[col_rate].max():.1f}%'),
    ('Lowest Rate (Pre)',     f'{df[df["Covid_Period"]=="Pre-Covid"][col_rate].min():.1f}%'),
    ('States Analysed',       f'{df["Region"].nunique()}'),
    ('Date Range',            '2019–2020'),
    ('Covid Impact (avg pp)', f'+{impact.mean():.1f}pp'),
]
DARK2 = "#1a1a2e"
clist = [PRE,COVID,POST,COVID,POST,DARK2,DARK2,COVID]
for i, (label, val) in enumerate(metrics):
    y_pos = 0.92 - i*0.12
    c = clist[i] if i < len(clist) else DARK2
    ax6.text(0.0, y_pos, label+":", fontsize=10, transform=ax6.transAxes, color="#555", va="center")
    ax6.text(1.0, y_pos, val, fontsize=11, fontweight="bold", transform=ax6.transAxes, color=c, va="center", ha="right")
    ax6.plot([0, 1], [y_pos-0.05, y_pos-0.05], color="#ddd", lw=0.5, transform=ax6.transAxes)
ax6.set_title('Key Metrics Summary', fontweight='bold', fontsize=10, pad=8)

DARK = '#1a1a2e'
plt.savefig(f'{PLOTS}/10_policy_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('  ✅ 10_policy_dashboard.png')


# ════════════════════════════════════════════════════════════════
# 12. SUMMARY STATISTICS
# ════════════════════════════════════════════════════════════════
print("\n── FINAL SUMMARY ──")
print(f"\n  Pre-Covid  Avg Rate : {df[df['Covid_Period']=='Pre-Covid'][col_rate].mean():.2f}%")
print(f"  Lockdown   Avg Rate : {df[df['Covid_Period']=='Lockdown'][col_rate].mean():.2f}%")
print(f"  Recovery   Avg Rate : {df[df['Covid_Period']=='Recovery'][col_rate].mean():.2f}%")
print(f"  Peak Rate            : {df[col_rate].max():.2f}%")
print(f"  Most Impacted State  : {impact.idxmax()} (+{impact.max():.1f}pp)")
print(f"  Fastest Recovery     : {recovery_speed.idxmax()} (-{recovery_speed.max():.1f}pp)")
print(f"\n  All 10 plots saved in: ../plots/")
print("="*65)
print("  ✅ ANALYSIS COMPLETE")
print("="*65)
