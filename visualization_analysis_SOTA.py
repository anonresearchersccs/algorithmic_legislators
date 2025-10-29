"""
High-Quality Visualization and Statistical Analysis for GIQ Publication
========================================================================

This module provides publication-ready figures and rigorous statistical analysis.

Key Features:
- Publication-quality plots with confidence intervals
- Statistical hypothesis testing (ANOVA, t-tests, effect sizes)
- Pareto frontier analysis
- Interaction and mediation analysis
- Professional formatting for academic journals

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import os
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# =============================================================================
# 1. Statistical Analysis Functions
# =============================================================================

def compute_effect_size_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Interpretation:
    - Small: d = 0.2
    - Medium: d = 0.5
    - Large: d = 0.8
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def perform_anova_analysis(df: pd.DataFrame, metric: str, 
                           factors: List[str]) -> Dict:
    """
    Perform ANOVA to test main effects and interactions.
    
    Returns dict with F-statistics, p-values, and effect sizes.
    """
    from scipy.stats import f_oneway
    
    results = {}
    
    # Main effects for each factor
    for factor in factors:
        groups = [group[metric].values for name, group in df.groupby(factor)]
        groups = [g for g in groups if len(g) > 0]  # Filter empty groups
        
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            results[factor] = {
                'F': f_stat,
                'p': p_val,
                'significant': p_val < 0.05
            }
    
    return results


def compute_treatment_effects(df: pd.DataFrame, baseline_condition: Dict,
                               metrics: List[str]) -> pd.DataFrame:
    """
    Compute treatment effects relative to baseline with statistical tests.
    
    Returns DataFrame with means, SDs, p-values, and effect sizes.
    """
    # Get baseline data
    baseline_query = " & ".join([f"{k} == {v}" for k, v in baseline_condition.items()])
    baseline = df.query(baseline_query)
    
    results = []
    
    # Get last step for each replicate
    df_last = df.groupby(['scenario_id', 'replicate']).tail(1)
    
    # Iterate through unique treatment combinations
    treatment_cols = ['ai_proportion', 'transparency', 'adapt_threshold']
    treatment_cols = [c for c in treatment_cols if c in df.columns]
    
    for treatment_vals, group in df_last.groupby(treatment_cols):
        if not isinstance(treatment_vals, tuple):
            treatment_vals = (treatment_vals,)
        
        treatment_dict = dict(zip(treatment_cols, treatment_vals))
        
        # Check if this is baseline
        is_baseline = all(treatment_dict.get(k) == v for k, v in baseline_condition.items())
        
        row = treatment_dict.copy()
        
        for metric in metrics:
            baseline_values = df_last.query(baseline_query)[metric].values
            treatment_values = group[metric].values
            
            if len(treatment_values) == 0 or len(baseline_values) == 0:
                continue
            
            # Statistics
            mean_val = np.mean(treatment_values)
            std_val = np.std(treatment_values, ddof=1)
            
            # T-test vs baseline
            if not is_baseline and len(baseline_values) > 1 and len(treatment_values) > 1:
                t_stat, p_val = stats.ttest_ind(treatment_values, baseline_values)
                effect_size = compute_effect_size_cohens_d(treatment_values, baseline_values)
            else:
                t_stat, p_val, effect_size = 0, 1.0, 0.0
            
            # Significance stars
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = ""
            
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
            row[f"{metric}_p"] = p_val
            row[f"{metric}_d"] = effect_size
            row[f"{metric}_sig"] = sig
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_confidence_intervals(df: pd.DataFrame, metric: str, 
                                 groupby_cols: List[str], 
                                 ci_level: float = 0.95) -> pd.DataFrame:
    """
    Compute confidence intervals for metric grouped by specified columns.
    
    Returns DataFrame with mean, CI_low, CI_high.
    """
    from scipy.stats import t as t_dist
    
    alpha = 1 - ci_level
    
    results = []
    for group_vals, group_data in df.groupby(groupby_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        
        values = group_data[metric].values
        n = len(values)
        
        if n < 2:
            continue
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        se = std / np.sqrt(n)
        
        # T-distribution critical value
        t_crit = t_dist.ppf(1 - alpha/2, n - 1)
        margin = t_crit * se
        
        result = dict(zip(groupby_cols, group_vals))
        result[f'{metric}_mean'] = mean
        result[f'{metric}_ci_low'] = mean - margin
        result[f'{metric}_ci_high'] = mean + margin
        result[f'{metric}_std'] = std
        result[f'{metric}_n'] = n
        
        results.append(result)
    
    return pd.DataFrame(results)


# =============================================================================
# 2. High-Quality Visualization Functions
# =============================================================================

def plot_tradeoff_analysis(df: pd.DataFrame, x_metric='system_performance', 
                           y_metric='trust', color_var='ai_proportion',
                           facet_var='transparency',
                           out_html='figure1_tradeoff.html',
                           out_png: Optional[str] = None) -> go.Figure:
    """
    Create publication-quality trade-off plot with Pareto frontier.
    
    This is Figure 1 in the manuscript.
    """
    # Get last step data
    df_last = df.groupby(['scenario_id', 'replicate']).tail(1)
    
    # Compute aggregated means
    group_cols = [color_var, facet_var] if facet_var else [color_var]
    df_agg = df_last.groupby(group_cols).agg({
        x_metric: ['mean', 'std'],
        y_metric: ['mean', 'std'],
        'discovered_inconsistency': 'mean'
    }).reset_index()
    
    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]
    
    # Create figure
    if facet_var:
        facet_vals = sorted(df_agg[facet_var].unique())
        n_facets = len(facet_vals)
        fig = make_subplots(
            rows=1, cols=n_facets,
            subplot_titles=[f"{facet_var.capitalize()}={v}" for v in facet_vals],
            horizontal_spacing=0.12
        )
        
        for i, fval in enumerate(facet_vals):
            df_facet = df_agg[df_agg[facet_var] == fval]
            
            # Add scatter with error bars
            for _, row in df_facet.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[row[f'{x_metric}_mean']],
                        y=[row[f'{y_metric}_mean']],
                        error_x=dict(type='data', array=[row[f'{x_metric}_std']]),
                        error_y=dict(type='data', array=[row[f'{y_metric}_std']]),
                        mode='markers',
                        marker=dict(
                            size=15,
                            color=row[color_var],
                            colorscale='Viridis',
                            line=dict(width=1, color='white'),
                            showscale=(i == n_facets - 1)
                        ),
                        name=f"AI={row[color_var]:.1%}",
                        showlegend=(i == 0),
                        hovertemplate=f"AI: {row[color_var]:.1%}<br>" +
                                     f"{x_metric}: {row[f'{x_metric}_mean']:.3f}<br>" +
                                     f"{y_metric}: {row[f'{y_metric}_mean']:.3f}<br>" +
                                     f"Discovered inconsistency: {row['discovered_inconsistency_mean']:.2f}<extra></extra>"
                    ),
                    row=1, col=i+1
                )
            
            # Add trend line
            x_vals = df_facet[f'{x_metric}_mean'].values
            y_vals = df_facet[f'{y_metric}_mean'].values
            
            if len(x_vals) > 1:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=p(x_trend),
                        mode='lines',
                        line=dict(dash='dash', color='gray'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=i+1
                )
            
            # Update axes
            fig.update_xaxes(title_text=x_metric.replace('_', ' ').title() if i == 1 else "",
                            row=1, col=i+1)
            fig.update_yaxes(title_text=y_metric.replace('_', ' ').title() if i == 0 else "",
                            row=1, col=i+1)
    
    else:
        # Single plot without facets
        fig = go.Figure()
        
        for _, row in df_agg.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row[f'{x_metric}_mean']],
                    y=[row[f'{y_metric}_mean']],
                    error_x=dict(type='data', array=[row[f'{x_metric}_std']]),
                    error_y=dict(type='data', array=[row[f'{y_metric}_std']]),
                    mode='markers',
                    marker=dict(size=15, color=row[color_var], colorscale='Viridis'),
                    name=f"AI={row[color_var]:.1%}"
                )
            )
        
        fig.update_xaxes(title_text=x_metric.replace('_', ' ').title())
        fig.update_yaxes(title_text=y_metric.replace('_', ' ').title())
    
    fig.update_layout(
        title="Trade-off Analysis: System Performance vs. Trust",
        height=400,
        width=1200 if facet_var else 600,
        template='plotly_white',
        font=dict(size=12),
        showlegend=True,
        margin=dict(l=60, r=30, t=50, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    fig.write_html(out_html)
    print(f"[SAVED] {out_html}")
    if out_png:
        os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
        fig.write_image(out_png, width=1200 if facet_var else 600, height=400, scale=2)
        print(f"[SAVED] {out_png}")
    
    return fig


def plot_temporal_dynamics(df: pd.DataFrame, scenarios_to_plot: List[int],
                           metrics: List[str] = ['trust', 'system_performance', 
                                                 'discovered_inconsistency', 'pass_rate'],
                           out_html='figure2_dynamics.html',
                           out_png: Optional[str] = None) -> go.Figure:
    """
    Plot temporal evolution of key metrics with confidence bands.
    
    This is Figure 2 in the manuscript.
    """
    # Filter scenarios
    df_filtered = df[df['scenario_id'].isin(scenarios_to_plot)]
    
    # Compute CI by scenario and step
    n_metrics = len(metrics)
    n_scenarios = len(scenarios_to_plot)
    
    fig = make_subplots(
        rows=n_metrics, cols=1,
        subplot_titles=[m.replace('_', ' ').title() for m in metrics],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    colors = px.colors.qualitative.Set2
    
    for i, metric in enumerate(metrics):
        for j, sc_id in enumerate(scenarios_to_plot):
            df_sc = df_filtered[df_filtered['scenario_id'] == sc_id]
            
            # Compute mean and CI by step
            stats_by_step = df_sc.groupby('step').agg({
                metric: ['mean', 'std', 'count']
            }).reset_index()
            
            stats_by_step.columns = ['step', 'mean', 'std', 'count']
            stats_by_step['se'] = stats_by_step['std'] / np.sqrt(stats_by_step['count'])
            stats_by_step['ci'] = 1.96 * stats_by_step['se']
            
            # Get scenario description
            sc_params = df_sc.iloc[0]
            sc_label = f"AI={sc_params['ai_proportion']:.0%}, T={sc_params['transparency']:.1f}"
            
            # Plot mean line
            fig.add_trace(
                go.Scatter(
                    x=stats_by_step['step'],
                    y=stats_by_step['mean'],
                    mode='lines',
                    name=sc_label,
                    line=dict(color=colors[j % len(colors)]),
                    showlegend=(i == 0),
                    legendgroup=f"sc{j}"
                ),
                row=i+1, col=1
            )
            
            # Add confidence band
            fig.add_trace(
                go.Scatter(
                    x=stats_by_step['step'].tolist() + stats_by_step['step'].tolist()[::-1],
                    y=(stats_by_step['mean'] + stats_by_step['ci']).tolist() + 
                      (stats_by_step['mean'] - stats_by_step['ci']).tolist()[::-1],
                    fill='toself',
                    fillcolor=colors[j % len(colors)],
                    opacity=0.2,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip',
                    legendgroup=f"sc{j}"
                ),
                row=i+1, col=1
            )
        
        # Update y-axis
        fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=i+1, col=1)
    
    # Update x-axis (only bottom)
    fig.update_xaxes(title_text="Simulation Step", row=n_metrics, col=1)
    
    fig.update_layout(
        title="Temporal Dynamics of Key Metrics",
        height=300 * n_metrics,
        width=1000,
        template='plotly_white',
        font=dict(size=11),
        hovermode='x unified',
        margin=dict(l=70, r=30, t=60, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    fig.write_html(out_html)
    print(f"[SAVED] {out_html}")
    if out_png:
        os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
        fig.write_image(out_png, width=1000, height=300 * n_metrics, scale=2)
        print(f"[SAVED] {out_png}")
    
    return fig


def plot_spatial_trajectories(df: pd.DataFrame, scenario_id: int = 1,
                              replicate: int = 0,
                              out_html='figure3_spatial.html',
                              out_png: Optional[str] = None) -> go.Figure:
    """
    Plot party position trajectories in 2D PCA space with annotations.
    
    This is Figure 3 in the manuscript.
    """
    # Filter data
    df_filtered = df[(df['scenario_id'] == scenario_id) & 
                     (df['replicate'] == replicate)]
    
    if len(df_filtered) == 0:
        print(f"[WARNING] No data for scenario {scenario_id}, replicate {replicate}")
        return None
    
    # Extract position data
    all_positions = []
    entity_data = {}  # {entity_name: [(step, vector), ...]}
    
    for _, row in df_filtered.iterrows():
        step = row['step']
        
        for p in row['party_positions']:
            party_name = p['party']
            vec = np.array(p['declared_vector'])
            
            if party_name not in entity_data:
                entity_data[party_name] = []
            entity_data[party_name].append((step, vec))
            all_positions.append(vec)
    
    # PCA if dimension > 2
    all_positions = np.array(all_positions)
    dimension = all_positions.shape[1]
    
    if dimension > 2:
        pca = PCA(n_components=2)
        pca.fit(all_positions)
        explained_var = pca.explained_variance_ratio_
        x_label = f"PC1 ({explained_var[0]:.1%} var.)"
        y_label = f"PC2 ({explained_var[1]:.1%} var.)"
    else:
        pca = None
        x_label = "Dimension 1"
        y_label = "Dimension 2"
    
    # Create figure
    fig = go.Figure()
    
    colors_party = px.colors.qualitative.Plotly
    
    for i, (entity_name, trajectory) in enumerate(entity_data.items()):
        # Sort by step
        trajectory.sort(key=lambda x: x[0])
        steps, vectors = zip(*trajectory)
        vectors = np.array(vectors)
        
        # Transform to 2D
        if pca is not None:
            vectors_2d = pca.transform(vectors)
        else:
            vectors_2d = vectors
        
        # Plot trajectory
        fig.add_trace(
            go.Scatter(
                x=vectors_2d[:, 0],
                y=vectors_2d[:, 1],
                mode='lines+markers',
                name=entity_name,
                line=dict(color=colors_party[i % len(colors_party)], width=2),
                marker=dict(size=6, symbol='circle'),
                hovertemplate=f"{entity_name}<br>Step: %{{text}}<extra></extra>",
                text=steps
            )
        )
        
        # Annotate start and end
        fig.add_annotation(
            x=vectors_2d[0, 0], y=vectors_2d[0, 1],
            text="Start",
            showarrow=False,
            font=dict(size=8, color=colors_party[i % len(colors_party)]),
            xshift=10, yshift=10
        )
        
        fig.add_annotation(
            x=vectors_2d[-1, 0], y=vectors_2d[-1, 1],
            text="End",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=colors_party[i % len(colors_party)],
            font=dict(size=8, color=colors_party[i % len(colors_party)]),
            ax=20, ay=-20
        )
    
    fig.update_layout(
        title=f"Party Position Trajectories (Scenario {scenario_id})",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        width=800,
        template='plotly_white',
        font=dict(size=12),
        showlegend=True,
        legend=dict(x=1.05, y=1),
        margin=dict(l=70, r=30, t=60, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    fig.write_html(out_html)
    print(f"[SAVED] {out_html}")
    if out_png:
        os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
        fig.write_image(out_png, width=800, height=600, scale=2)
        print(f"[SAVED] {out_png}")
    
    return fig


def plot_interaction_effects(df: pd.DataFrame, metric='system_performance',
                             x_var='ai_proportion', line_var='transparency',
                             out_html='figure4_interaction.html',
                             out_png: Optional[str] = None) -> go.Figure:
    """
    Plot interaction effects between AI proportion and transparency.
    
    This is Figure 4 in the manuscript.
    """
    # Get last step
    df_last = df.groupby(['scenario_id', 'replicate']).tail(1)
    
    # Compute means and CIs
    df_stats = df_last.groupby([x_var, line_var]).agg({
        metric: ['mean', 'std', 'count']
    }).reset_index()
    
    df_stats.columns = [x_var, line_var, 'mean', 'std', 'count']
    df_stats['se'] = df_stats['std'] / np.sqrt(df_stats['count'])
    df_stats['ci'] = 1.96 * df_stats['se']
    
    fig = go.Figure()
    
    line_vals = sorted(df_stats[line_var].unique())
    colors = px.colors.qualitative.Safe
    
    for i, lval in enumerate(line_vals):
        df_line = df_stats[df_stats[line_var] == lval].sort_values(x_var)
        
        fig.add_trace(
            go.Scatter(
                x=df_line[x_var],
                y=df_line['mean'],
                error_y=dict(type='data', array=df_line['ci']),
                mode='lines+markers',
                name=f"{line_var.capitalize()}={lval}",
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=10)
            )
        )
    
    fig.update_layout(
        title=f"Interaction Effect: {x_var.replace('_', ' ').title()} × " + 
              f"{line_var.replace('_', ' ').title()}",
        xaxis_title=x_var.replace('_', ' ').title(),
        yaxis_title=metric.replace('_', ' ').title(),
        height=500,
        width=700,
        template='plotly_white',
        font=dict(size=12),
        showlegend=True,
        margin=dict(l=70, r=30, t=60, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    fig.write_html(out_html)
    print(f"[SAVED] {out_html}")
    if out_png:
        os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
        fig.write_image(out_png, width=700, height=500, scale=2)
        print(f"[SAVED] {out_png}")
    
    return fig


def create_results_table(df: pd.DataFrame, baseline_condition: Dict,
                         metrics: List[str],
                         out_csv='table1_results.csv') -> pd.DataFrame:
    """
    Create main results table with statistical tests.
    
    This is Table 1 in the manuscript.
    """
    results = compute_treatment_effects(df, baseline_condition, metrics)
    
    # Format for publication
    table_rows = []
    
    for _, row in results.iterrows():
        # Create treatment description
        ai_prop = row.get('ai_proportion', 0)
        transp = row.get('transparency', 0)
        adapt_thr = row.get('adapt_threshold', 1.0)
        
        treatment_desc = f"AI={ai_prop:.0%}, T={transp:.1f}"
        if 'adapt_threshold' in row:
            treatment_desc += f", Adapt={adapt_thr:.1f}"
        
        row_dict = {'Treatment': treatment_desc}
        
        for metric in metrics:
            mean_val = row.get(f'{metric}_mean', np.nan)
            std_val = row.get(f'{metric}_std', np.nan)
            sig = row.get(f'{metric}_sig', '')
            d = row.get(f'{metric}_d', np.nan)
            
            # Format as "M (SD) sig"
            formatted = f"{mean_val:.3f} ({std_val:.3f}){sig}"
            row_dict[metric.replace('_', ' ').title()] = formatted
            row_dict[f"{metric}_d"] = f"{d:.2f}" if not np.isnan(d) else ""
        
        table_rows.append(row_dict)
    
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")
    
    return table_df


def compute_pareto_frontier(df: pd.DataFrame, metric1='system_performance',
                           metric2='trust') -> pd.DataFrame:
    """
    Compute Pareto frontier for two objectives.
    
    Returns DataFrame with Pareto-optimal points.
    """
    # Get last step aggregated by scenario
    df_last = df.groupby(['scenario_id', 'replicate']).tail(1)
    df_agg = df_last.groupby('scenario_id').agg({
        metric1: 'mean',
        metric2: 'mean',
        'ai_proportion': 'first',
        'transparency': 'first'
    }).reset_index()
    
    # Find Pareto frontier (maximize both)
    pareto_mask = np.ones(len(df_agg), dtype=bool)
    
    for i in range(len(df_agg)):
        for j in range(len(df_agg)):
            if i != j:
                # Check if j dominates i
                if (df_agg.iloc[j][metric1] >= df_agg.iloc[i][metric1] and
                    df_agg.iloc[j][metric2] >= df_agg.iloc[i][metric2] and
                    (df_agg.iloc[j][metric1] > df_agg.iloc[i][metric1] or
                     df_agg.iloc[j][metric2] > df_agg.iloc[i][metric2])):
                    pareto_mask[i] = False
                    break
    
    pareto_points = df_agg[pareto_mask].sort_values(metric1)
    
    return pareto_points


# =============================================================================
# 3. Summary Statistics
# =============================================================================

def generate_summary_statistics(df: pd.DataFrame, 
                                out_csv='summary_statistics_SOTA.csv') -> pd.DataFrame:
    """
    Generate comprehensive summary statistics by scenario.
    """
    # Get last step
    df_last = df.groupby(['scenario_id', 'replicate']).tail(1)
    
    # Group by scenario
    summary = df_last.groupby(['scenario_id', 'ai_proportion', 'transparency']).agg({
        'trust': ['mean', 'std', 'min', 'max'],
        'system_performance': ['mean', 'std', 'min', 'max'],
        'voter_satisfaction': ['mean', 'std', 'min', 'max'],
        'latent_inconsistency': ['mean', 'std', 'min', 'max'],
        'discovered_inconsistency': ['mean', 'std', 'min', 'max'],
        'pass_rate': ['mean', 'std', 'min', 'max'],
        'representativity': ['mean', 'std', 'min', 'max'],
        'replicate': 'count'
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary = summary.rename(columns={'replicate_count': 'n_replicates'})
    
    summary.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")
    
    return summary


print("[INFO] Visualization and analysis functions loaded successfully.")

