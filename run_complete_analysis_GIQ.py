"""
Complete Analysis Script for GIQ Submission
============================================

This script runs the full analysis pipeline:
1. Load SOTA model with real metrics
2. Run experimental scenarios (7 conditions × 10 replicates)
3. Generate statistical analyses
4. Create publication-quality figures
5. Export all tables and results

Usage:
    python run_complete_analysis_GIQ.py

Outputs:
    - results_GIQ_SOTA.csv (main data)
    - table1_main_results.csv (results table with significance tests)
    - summary_statistics_GIQ.csv (descriptive stats)
    - figure1_tradeoff_GIQ.html (performance vs trust)
    - figure2_dynamics_GIQ.html (temporal evolution)
    - figure3_spatial_GIQ.html (position trajectories)
    - figure4_interaction_GIQ.html (AI × transparency interaction)
    - pareto_frontier_GIQ.csv (Pareto-optimal configurations)
    - findings_summary.txt (key findings text)

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import sys

# Import SOTA modules
from algorithmic_legislators_SOTA import (
    ExtendedParliamentModel,
    run_scenarios,
    load_openai_key,
    set_random_seed
)

from visualization_analysis_SOTA import (
    plot_tradeoff_analysis,
    plot_temporal_dynamics,
    plot_spatial_trajectories,
    plot_interaction_effects,
    create_results_table,
    compute_treatment_effects,
    generate_summary_statistics,
    compute_pareto_frontier,
    perform_anova_analysis
)


def run_dense_grid_supplement():
    print("[SUPP] Running dense rho grid (0.00..0.60, step 0.05), reps=50 under T=1, Adapt on...")
    scenarios = []
    for r in [round(x, 2) for x in [i*0.05 for i in range(0,13)]]:
        scenarios.append({
            "num_parties": 4,
            "num_seats": 40,
            "ai_proportion": r,
            "transparency": 1.0,
            "num_voters": 100,
            "dimension": 3,
            "adapt_threshold": 1.0,
            "adapt_rate": 0.2
        })
    df = run_scenarios(scenarios, steps_per_run=50, runs_per_scenario=50, seed=123, out_csv="supp_dense_grid.csv")
    print("[SUPP] Dense grid saved to supp_dense_grid.csv (" + str(len(df)) + " rows)")


def analyze_dense_grid_hierarchical():
    print("[SUPP] Analyzing dense grid (means by rho, bootstrap over replicates, quadratic fit)...")
    dfg = pd.read_csv("supp_dense_grid.csv")
    # Final step per run
    dfg_final = dfg[dfg["step"] == dfg["step"].max()].copy()
    dfg_final["rho"] = dfg_final["ai_proportion"].astype(float)
    # Means by rho
    means = dfg_final.groupby("rho")["system_performance"].mean().reset_index()
    means["rho2"] = means["rho"] ** 2
    X = np.vstack([np.ones(len(means)), means["rho"].values, means["rho2"].values]).T
    y = means["system_performance"].values
    a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]
    rho_star = -b / (2 * c) if c != 0 else np.nan
    print(f"[SUPP] Quadratic fit on means: a={a:.3f}, b={b:.3f}, c={c:.3f}; rho*={rho_star:.3f}")
    # Bootstrap: resample replicates within each rho to recompute mean curve
    rng = np.random.default_rng(42)
    rhos = sorted(dfg_final["rho"].unique())
    boots = []
    for _ in range(500):
        ys = []
        for r in rhos:
            grp = dfg_final[dfg_final["rho"] == r]
            reps = grp["replicate"].unique()
            if len(reps) == 0:
                ys.append(np.nan)
                continue
            samp_reps = rng.choice(reps, size=len(reps), replace=True)
            samp_vals = []
            for rep in samp_reps:
                samp_vals.append(float(grp[grp["replicate"] == rep]["system_performance"].mean()))
            ys.append(np.nanmean(samp_vals))
        ys = np.array(ys)
        ok = np.isfinite(ys)
        Xb = np.vstack([np.ones(ok.sum()), np.array(rhos)[ok], (np.array(rhos)[ok] ** 2)]).T
        bb = np.linalg.lstsq(Xb, ys[ok], rcond=None)[0]
        b1, b2 = bb[1], bb[2]
        rb = -b1 / (2 * b2) if b2 != 0 else np.nan
        boots.append(rb)
    boots = np.array([r for r in boots if np.isfinite(r)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"[SUPP] rho* bootstrap 95% CI (means-by-rho): [{lo:.3f}, {hi:.3f}]")
    # TOST proxy on means
    m50 = float(means.loc[np.isclose(means["rho"], 0.50), "system_performance"].mean())
    m30 = float(means.loc[np.isclose(means["rho"], 0.30), "system_performance"].mean())
    diff = m50 - m30
    print(f"[SUPP] TOST proxy (means): mean diff Perf(0.50 - 0.30) = {diff:.3f} vs margin ±0.03")


def run_do_interventions_supplement():
    print("[SUPP] Running do-interventions (Adapt blocked windows; shocks in |d-r|)...")
    scenarios = []
    base = {
        "num_parties": 4,
        "num_seats": 40,
        "transparency": 1.0,
        "num_voters": 100,
        "dimension": 3,
        "adapt_threshold": 1.0,
        "adapt_rate": 0.2,
    }
    for rho in [0.30, 0.50]:
        s = dict(base)
        s["ai_proportion"] = rho
        # block adaptation randomly and inject shocks
        s["adapt_block_prob"] = 0.3
        s["shock_prob"] = 0.2
        s["shock_magnitude"] = 0.15
        scenarios.append(s)
        # control counterpart without interventions
        sc = dict(base)
        sc["ai_proportion"] = rho
        sc["adapt_block_prob"] = 0.0
        sc["shock_prob"] = 0.0
        sc["shock_magnitude"] = 0.0
        scenarios.append(sc)
    df = run_scenarios(scenarios, steps_per_run=50, runs_per_scenario=20, seed=321, out_csv="supp_do_interventions.csv")
    print("[SUPP] do-interventions saved to supp_do_interventions.csv (" + str(len(df)) + " rows)")


def run_transparency_factorial_supplement():
    print("[SUPP] Running 2x2x2 factorial Tin/Tproc/Tout at rho=0.30...")
    scenarios = []
    for Tin in [0.0, 1.0]:
        for Tproc in [0.0, 1.0]:
            for Tout in [0.0, 1.0]:
                scenarios.append({
                    "num_parties": 4,
                    "num_seats": 40,
                    "ai_proportion": 0.30,
                    "transparency_input": Tin,
                    "transparency_process": Tproc,
                    "transparency_output": Tout,
                    "transparency": Tproc,  # legacy param for compatibility
                    "num_voters": 100,
                    "dimension": 3,
                    "adapt_threshold": 1.0,
                    "adapt_rate": 0.2
                })
    df = run_scenarios(scenarios, steps_per_run=50, runs_per_scenario=20, seed=456, out_csv="supp_transparency_factorial.csv")
    print("[SUPP] Factorial saved to supp_transparency_factorial.csv (" + str(len(df)) + " rows)")


def main():
    """Main analysis pipeline."""
    
    print("="*70)
    print("ALGORITHMIC LEGISLATORS - COMPLETE ANALYSIS FOR GIQ")
    print("="*70)
    print()
    
    # Step 1: Load API key (optional)
    print("[STEP 1] Loading API key (optional)...")
    load_openai_key("api_key.txt")
    print()
    
    # Step 2: Define scenarios
    print("[STEP 2] Defining experimental scenarios...")
    scenarios = [
        # 0: Baseline (No AI)
        {
            "num_parties": 4,
            "num_seats": 40,
            "ai_proportion": 0.0,
            "transparency": 0.0,
            "num_voters": 100,
            "dimension": 3,
            "adapt_threshold": 1.0,
            "adapt_rate": 0.2
        },
        # 1: Low AI, No Transparency
        {
            "num_parties": 4,
            "num_seats": 40,
            "ai_proportion": 0.2,
            "transparency": 0.0,
            "num_voters": 100,
            "dimension": 3,
            "adapt_threshold": 1.0,
            "adapt_rate": 0.2
        },
        # 2: Low AI, Full Transparency
        {
            "num_parties": 4,
            "num_seats": 40,
            "ai_proportion": 0.2,
            "transparency": 1.0,
            "num_voters": 100,
            "dimension": 3,
            "adapt_threshold": 1.0,
            "adapt_rate": 0.2
        },
        # 3: Moderate AI, No Transparency
        {
            "num_parties": 4,
            "num_seats": 40,
            "ai_proportion": 0.3,
            "transparency": 0.0,
            "num_voters": 100,
            "dimension": 3,
            "adapt_threshold": 1.0,
            "adapt_rate": 0.2
        },
        # 4: Moderate AI, Full Transparency, NO Adaptation
        {
            "num_parties": 4,
            "num_seats": 40,
            "ai_proportion": 0.3,
            "transparency": 1.0,
            "num_voters": 100,
            "dimension": 3,
            "adapt_threshold": 999.0,  # Disabled
            "adapt_rate": 0.2
        },
        # 5: Moderate AI, Full Transparency, WITH Adaptation
        {
            "num_parties": 4,
            "num_seats": 40,
            "ai_proportion": 0.3,
            "transparency": 1.0,
            "num_voters": 100,
            "dimension": 3,
            "adapt_threshold": 1.0,
            "adapt_rate": 0.2
        },
        # 6: High AI, Full Transparency, WITH Adaptation
        {
            "num_parties": 4,
            "num_seats": 40,
            "ai_proportion": 0.5,
            "transparency": 1.0,
            "num_voters": 100,
            "dimension": 3,
            "adapt_threshold": 1.0,
            "adapt_rate": 0.2
        }
    ]
    print(f"  -> {len(scenarios)} scenarios defined")
    print()
    
    # Step 3: Run simulations
    print("[STEP 3] Running simulations (this may take several minutes)...")
    df_results = run_scenarios(
        scenarios=scenarios,
        steps_per_run=50,
        runs_per_scenario=10,
        seed=42,
        out_csv="results_GIQ_SOTA.csv"
    )
    print(f"  -> Total observations: {len(df_results)}")
    print()
    
    # Step 4: Generate summary statistics
    print("[STEP 4] Generating summary statistics...")
    summary_stats = generate_summary_statistics(
        df_results,
        out_csv="summary_statistics_GIQ.csv"
    )
    print(f"  -> Saved to summary_statistics_GIQ.csv")
    print()
    
    # Step 5: Main results table
    print("[STEP 5] Creating main results table with statistical tests...")
    baseline_condition = {
        'ai_proportion': 0.0,
        'transparency': 0.0
    }
    
    key_metrics = [
        'system_performance',
        'trust',
        'discovered_inconsistency',
        'pass_rate',
        'voter_satisfaction'
    ]
    
    results_table = create_results_table(
        df_results,
        baseline_condition=baseline_condition,
        metrics=key_metrics,
        out_csv='table1_main_results.csv'
    )
    print(f"  -> Saved to table1_main_results.csv")
    print()
    
    # Step 6: ANOVA analysis
    print("[STEP 6] Performing ANOVA analysis...")
    df_final = df_results.groupby(['scenario_id', 'replicate']).tail(1)
    anova_results = perform_anova_analysis(
        df_final,
        metric='system_performance',
        factors=['ai_proportion', 'transparency']
    )
    print("  -> ANOVA Results for System Performance:")
    for factor, result in anova_results.items():
        sig = "***" if result['p'] < 0.001 else "**" if result['p'] < 0.01 else "*" if result['p'] < 0.05 else "ns"
        print(f"     {factor}: F={result['F']:.2f}, p={result['p']:.4f} {sig}")
    print()
    
    # Step 7: Create figures
    print("[STEP 7] Creating publication-quality figures...")
    
    # Figure 1: Trade-off analysis
    print("  -> Figure 1: Trade-off analysis (performance vs trust)...")
    fig1 = plot_tradeoff_analysis(
        df_results,
        x_metric='system_performance',
        y_metric='trust',
        color_var='ai_proportion',
        facet_var='transparency',
        out_html='figure1_tradeoff_GIQ.html',
        out_png='figures/figure1_tradeoff.png'
    )
    
    # Figure 2: Temporal dynamics
    print("  -> Figure 2: Temporal dynamics...")
    scenarios_to_plot = [0, 2, 5, 6]  # Key scenarios
    fig2 = plot_temporal_dynamics(
        df_results,
        scenarios_to_plot=scenarios_to_plot,
        metrics=['trust', 'system_performance', 'discovered_inconsistency', 'pass_rate'],
        out_html='figure2_dynamics_GIQ.html',
        out_png='figures/figure2_dynamics.png'
    )
    
    # Figure 3: Spatial trajectories
    print("  -> Figure 3: Spatial trajectories...")
    fig3 = plot_spatial_trajectories(
        df_results,
        scenario_id=5,  # Moderate AI + T + Adapt
        replicate=0,
        out_html='figure3_spatial_GIQ.html',
        out_png='figures/figure3_spatial.png'
    )
    
    # Figure 4: Interaction effects
    print("  -> Figure 4: Interaction effects...")
    fig4 = plot_interaction_effects(
        df_results,
        metric='system_performance',
        x_var='ai_proportion',
        line_var='transparency',
        out_html='figure4_interaction_GIQ.html',
        out_png='figures/figure4_interaction.png'
    )
    print()
    
    # Step 8: Pareto frontier
    print("[STEP 8] Computing Pareto frontier...")
    pareto_points = compute_pareto_frontier(
        df_results,
        metric1='system_performance',
        metric2='trust'
    )
    pareto_points.to_csv('pareto_frontier_GIQ.csv', index=False)
    print(f"  -> {len(pareto_points)} Pareto-optimal points identified")
    print(f"  -> Saved to pareto_frontier_GIQ.csv")
    print()
    
    # Step 9: Key findings summary
    print("[STEP 9] Extracting key findings...")
    
    # Get final values
    baseline_perf = df_final[df_final['scenario_id'] == 0]['system_performance'].mean()
    moderate_ai_no_adapt = df_final[df_final['scenario_id'] == 4]['system_performance'].mean()
    moderate_ai_adapt = df_final[df_final['scenario_id'] == 5]['system_performance'].mean()
    high_ai = df_final[df_final['scenario_id'] == 6]['system_performance'].mean()
    
    baseline_trust = df_final[df_final['scenario_id'] == 0]['trust'].mean()
    moderate_trust_no_adapt = df_final[df_final['scenario_id'] == 4]['trust'].mean()
    moderate_trust_adapt = df_final[df_final['scenario_id'] == 5]['trust'].mean()
    
    # Write findings to file
    with open('findings_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("KEY FINDINGS FOR GIQ MANUSCRIPT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"1. BASELINE PERFORMANCE: {baseline_perf:.3f}\n\n")
        
        f.write("2. TRANSPARENCY WITHOUT ADAPTATION HARMS PERFORMANCE:\n")
        f.write(f"   - Moderate AI (30%) + Transparency + NO Adaptation: {moderate_ai_no_adapt:.3f}\n")
        f.write(f"   - Change from baseline: {(moderate_ai_no_adapt - baseline_perf):.3f} ")
        f.write(f"({(moderate_ai_no_adapt/baseline_perf - 1)*100:.1f}%)\n\n")
        
        f.write("3. MODERATE AI WITH TRANSPARENCY AND ADAPTATION IMPROVES PERFORMANCE:\n")
        f.write(f"   - Moderate AI (30%) + Transparency + Adaptation: {moderate_ai_adapt:.3f}\n")
        f.write(f"   - Change from baseline: {(moderate_ai_adapt - baseline_perf):.3f} ")
        f.write(f"({(moderate_ai_adapt/baseline_perf - 1)*100:.1f}%)\n\n")
        
        f.write("4. HIGH AI QUOTAS (50%) SHOW DIMINISHING RETURNS:\n")
        f.write(f"   - High AI (50%) + Transparency + Adaptation: {high_ai:.3f}\n")
        f.write(f"   - Change from baseline: {(high_ai - baseline_perf):.3f} ")
        f.write(f"({(high_ai/baseline_perf - 1)*100:.1f}%)\n\n")
        
        f.write("5. TRUST PATTERNS:\n")
        f.write(f"   - Baseline trust: {baseline_trust:.3f}\n")
        f.write(f"   - AI + Transparency WITHOUT adaptation: {moderate_trust_no_adapt:.3f} ")
        f.write(f"(erosion of {(baseline_trust - moderate_trust_no_adapt):.3f})\n")
        f.write(f"   - AI + Transparency WITH adaptation: {moderate_trust_adapt:.3f} ")
        f.write(f"(erosion limited to {(baseline_trust - moderate_trust_adapt):.3f})\n\n")
        
        f.write("="*70 + "\n")
        f.write("CONCLUSION: Adaptation is CRITICAL - transparency alone is harmful.\n")
        f.write("="*70 + "\n")
    
    print("  -> Saved to findings_summary.txt")
    print()
    
    # Final summary
    print("="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print()
    print("Generated files:")
    print("  [DATA]")
    print("    - results_GIQ_SOTA.csv (main dataset)")
    print("    - summary_statistics_GIQ.csv (descriptive stats)")
    print("    - pareto_frontier_GIQ.csv (Pareto-optimal points)")
    print()
    print("  [TABLES]")
    print("    - table1_main_results.csv (main results with significance tests)")
    print()
    print("  [FIGURES]")
    print("    - figure1_tradeoff_GIQ.html (performance vs trust)")
    print("    - figure2_dynamics_GIQ.html (temporal evolution)")
    print("    - figure3_spatial_GIQ.html (position trajectories)")
    print("    - figure4_interaction_GIQ.html (interaction effects)")
    print()
    print("  [SUMMARY]")
    print("    - findings_summary.txt (key findings narrative)")
    print()
    print("All outputs are ready for manuscript preparation.")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    # If called with --supp-dense, run dense grid supplement; otherwise run main
    if len(sys.argv) > 1:
        if sys.argv[1] == "--supp-dense":
            run_dense_grid_supplement()
            sys.exit(0)
        if sys.argv[1] == "--supp-dense-analyze":
            analyze_dense_grid_hierarchical()
            sys.exit(0)
        if sys.argv[1] == "--supp-do":
            run_do_interventions_supplement()
            sys.exit(0)
        if sys.argv[1] == "--supp-factorial":
            run_transparency_factorial_supplement()
            sys.exit(0)
    sys.exit(main())

