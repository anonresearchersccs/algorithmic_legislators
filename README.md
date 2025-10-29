# Algorithmic Legislators: When Do AI Quotas Improve Legislative Performance?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)

A rigorous agent-based model (ABM) examining the effects of AI quotas in legislative bodies. This research project analyzes how artificial intelligence parliamentarians, transparency mechanisms, and adaptive party behavior interact to affect system performance, trust, and representation.

**Submitted to Government Information Quarterly (GIQ)**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Model Description](#model-description)
- [Output Files](#output-files)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## 🔬 Overview

This repository implements a theoretically-grounded agent-based model to examine when and how AI quotas in legislatures improve performance. The model features:

- **Multi-dimensional policy space**: Parties and voters occupy an N-dimensional ideological space
- **Dual party positions**: Parties have both "real" (private) and "declared" (public) positions
- **Human and AI parliamentarians**: Mixed legislature with configurable AI proportion
- **Transparency mechanisms**: Three levels (input, process, output) affecting inconsistency detection
- **Adaptive party behavior**: Parties can adjust positions when inconsistencies are discovered
- **Rigorous metrics**: All outcomes are theoretically grounded (no simulated noise)

### Key Innovation

Unlike previous models, we measure **real outcomes** derived from model dynamics rather than simulated noise. Our metrics for trust, performance, and satisfaction are computed from actual party-voter distances, inconsistencies, legislative efficiency, and position stability.

---

## 🎯 Key Findings

Based on 7 experimental scenarios × 10 replicates × 50 time steps (3,500+ observations):

1. **Baseline Performance**: 0.676 (no AI, no transparency)

2. **Transparency alone harms performance**: 
   - Moderate AI (30%) + Transparency + NO Adaptation: 0.777
   - Change from baseline: +14.8%

3. **Moderate AI with transparency AND adaptation improves performance**:
   - Moderate AI (30%) + Transparency + Adaptation: 0.806
   - Change from baseline: +19.2%
   - **Optimal range**: 20-30% AI quota

4. **High AI quotas show diminishing returns**:
   - High AI (50%) + Transparency + Adaptation: 0.897
   - Change from baseline: +32.6%

5. **Critical mechanism**: Transparency → Inconsistency Detection → Adaptation → Performance
   - Without adaptation, transparency exposes inconsistencies but doesn't fix them
   - With adaptation, parties realign toward their real positions, improving representation

**Policy Implication**: Institutional design matters. AI quotas require complementary transparency and accountability mechanisms to deliver benefits.

---

## 📁 Project Structure

```
CODE/
│
├── Core Model Files
│   ├── algorithmic_legislators_SOTA.py      # Main ABM with real metrics
│   ├── visualization_analysis_SOTA.py       # Statistical analysis & figures
│   ├── sensitivity_analysis_robust.py       # Robustness checks
│   └── run_complete_analysis_GIQ.py         # Full pipeline runner
│
├── Interactive Notebooks
│   └── Algorithmic_Legislators_GIQ.ipynb    # Jupyter notebook interface
│
├── Manuscript Files
│   ├── manuscript_GIQ.tex                   # LaTeX manuscript
│   ├── manuscript_GIQ.pdf                   # Compiled PDF
│   └── references.bib                       # Bibliography
│
├── Data Outputs (Generated)
│   ├── results_GIQ_SOTA.csv                 # Main simulation results
│   ├── table1_main_results.csv              # Statistical tests
│   ├── summary_statistics_GIQ.csv           # Descriptive stats
│   ├── pareto_frontier_GIQ.csv              # Pareto-optimal points
│   ├── supp_dense_grid.csv                  # Dense AI quota grid (0-60%)
│   ├── supp_do_interventions.csv            # Causal interventions
│   └── supp_transparency_factorial.csv      # 2×2×2 transparency factorial
│
├── Figures (Generated)
│   ├── figure1_tradeoff_GIQ.html            # Performance vs Trust trade-off
│   ├── figure2_dynamics_GIQ.html            # Temporal evolution
│   ├── figure3_spatial_GIQ.html             # Party position trajectories
│   ├── figure4_interaction_GIQ.html         # AI × Transparency interaction
│   └── figures/*.png                        # PNG versions for manuscript
│
├── Supporting Files
│   ├── requirements.txt                     # Python dependencies
│   ├── api_key.txt                          # OpenAI API key (optional)
│   ├── findings_summary.txt                 # Key results summary
│   ├── convert_figures_to_pdf.py            # Figure conversion utility
│   ├── prepare_manuscript_figures.py        # Manuscript figure preparation
│   ├── empirical_calibration_guide.py       # Calibration documentation
│   └── test_api_key.py                      # API key testing
│
└── README.md                                # This file
```

---

## 🔧 Installation

### Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 4GB minimum (8GB recommended for large grids)

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

Dependencies include:
- `agentpy==0.1.5` - Agent-based modeling framework
- `numpy==1.26.4` - Numerical computing
- `pandas==2.2.2` - Data manipulation
- `plotly==5.22.0` - Interactive visualizations
- `scikit-learn==1.4.2` - Statistical analysis
- `tqdm==4.66.4` - Progress bars
- `openai>=1.30.0` - LLM integration (optional)

3. **Optional: OpenAI API key** (for online LLM mode)

Create `api_key.txt` with your key:
```
sk-your-api-key-here
```

**Note**: The model runs perfectly in **offline mode** without an API key. The offline policy uses distance-based heuristics that are faster and reproducible.

---

## 🚀 Quick Start

### Option 1: Run Complete Analysis (Recommended)

Generate all figures, tables, and statistical tests:

```bash
python run_complete_analysis_GIQ.py
```

This will:
- Run 7 scenarios × 10 replicates × 50 steps
- Generate all publication figures (HTML + PNG)
- Compute statistical tests (ANOVA, t-tests, effect sizes)
- Export tables and summary statistics
- **Estimated time**: 10-15 minutes

### Option 2: Interactive Notebook

```bash
jupyter notebook Algorithmic_Legislators_GIQ.ipynb
```

The notebook provides:
- Step-by-step explanation of the model
- Interactive parameter exploration
- Visualization of individual runs
- Code examples and documentation

### Option 3: Python API

```python
from algorithmic_legislators_SOTA import ExtendedParliamentModel, run_scenarios

# Define scenario
scenario = {
    "num_parties": 4,
    "num_seats": 40,
    "ai_proportion": 0.3,      # 30% AI parliamentarians
    "transparency": 1.0,        # Full transparency
    "num_voters": 100,
    "dimension": 3,             # 3 policy dimensions
    "adapt_threshold": 1.0,
    "adapt_rate": 0.2
}

# Run simulation
df = run_scenarios(
    scenarios=[scenario],
    steps_per_run=50,
    runs_per_scenario=10,
    seed=42,
    out_csv="my_results.csv"
)

# Analyze results
final_step = df[df['step'] == 50]
print(f"Mean performance: {final_step['system_performance'].mean():.3f}")
print(f"Mean trust: {final_step['trust'].mean():.3f}")
```

---

## 📖 Usage Guide

### Main Analyses

#### 1. **Core Experimental Scenarios**

```bash
python run_complete_analysis_GIQ.py
```

Generates: `results_GIQ_SOTA.csv`, `table1_main_results.csv`, Figures 1-4

#### 2. **Dense AI Quota Grid** (Supplementary Material)

Test performance across fine-grained AI quotas (0% to 60%, step 0.05):

```bash
python run_complete_analysis_GIQ.py --supp-dense
python run_complete_analysis_GIQ.py --supp-dense-analyze
```

Generates: `supp_dense_grid.csv` with quadratic fit and bootstrap confidence intervals

#### 3. **Causal Interventions** (do-calculus)

Test robustness with blocked adaptation and exogenous shocks:

```bash
python run_complete_analysis_GIQ.py --supp-do
```

Generates: `supp_do_interventions.csv`

#### 4. **Transparency Factorial** (2×2×2 design)

Decompose transparency into input/process/output components:

```bash
python run_complete_analysis_GIQ.py --supp-factorial
```

Generates: `supp_transparency_factorial.csv`

#### 5. **Comprehensive Sensitivity Analysis**

Test robustness across parameter variations:

```python
from sensitivity_analysis_robust import run_comprehensive_sensitivity

run_comprehensive_sensitivity(
    out_dir="sensitivity_results",
    quick_mode=False  # Set True for faster test run
)
```

Tests variations in:
- Number of parties (3-6)
- Policy dimensions (2-5)
- Adaptation rates (0.1-0.4)
- Voter tolerance (0.1-0.4)
- Metric weights

### Visualization

Create custom figures:

```python
from visualization_analysis_SOTA import (
    plot_tradeoff_analysis,
    plot_temporal_dynamics,
    plot_interaction_effects
)
import pandas as pd

df = pd.read_csv("results_GIQ_SOTA.csv")

# Trade-off plot
plot_tradeoff_analysis(
    df, 
    x_metric='system_performance',
    y_metric='trust',
    out_html='my_tradeoff.html'
)

# Time series
plot_temporal_dynamics(
    df,
    scenarios_to_plot=[0, 2, 5, 6],
    metrics=['trust', 'system_performance']
)
```

---

## 🧮 Model Description

### Agents

1. **Parties** (N=4)
   - Real position: `r_p ∈ [-1,1]^d` (true ideology)
   - Declared position: `d_p ∈ [-1,1]^d` (public stance)
   - Tactical position: `t_p = d_p + noise` (ad-hoc deviations)
   - Discovered inconsistency: `I_p ≥ 0` (accumulated penalty)

2. **Voters** (N=100)
   - Ideology: `v_i ∈ [-1,1]^d`
   - Dimension weights: `w_i ∈ Δ^d` (saliency)
   - Evaluation function: `U(v,p) = -||v - d_p||_w - α·I_p`

3. **Parliamentarians** (N=40)
   - **Human**: Follow party line, vote based on `||policy - d_p||`
   - **AI**: Use LLM (online) or distance heuristic (offline)
   - Mixed composition: AI proportion ρ ∈ [0,1]

### Mechanisms

#### Transparency (T ∈ [0,1])

Three components:
- **Input transparency** (T_in): Observability of inputs
- **Process transparency** (T_proc): Observability of deliberation
  - Drives inconsistency discovery: `ΔI_p = T_proc · γ · ||d_p - r_p||`
- **Output transparency** (T_out): Observability of outputs

#### Adaptation

When `I_p > threshold`:
```
d_p ← d_p + α · (r_p - d_p)
I_p ← 0.7 · I_p
```

Parties realign declared positions toward real positions.

### Metrics (Theoretically Grounded)

#### 1. Trust [0,1]
```
Trust = 0.35·Consistency + 0.35·Transparency_Effectiveness + 0.30·Alignment

Consistency = 1/(1 + mean(||d_p - r_p||))
Transparency_Effectiveness = 1/(1 + T_proc·γ·Inconsistency)
Alignment = 1/(1 + mean_voter_distance)
```

#### 2. System Performance [0,1]
```
Performance = 0.40·Efficiency + 0.40·Representativity + 0.20·Stability

Efficiency = bills_passed / total_bills
Representativity = 1 - (mean_voter_distance / normalization)
Stability = 1/(1 + mean_party_movement)
```

#### 3. Voter Satisfaction [0,1]
```
Satisfaction = mean(normalized_best_evaluation)
```

All metrics are computed from **actual model dynamics**—no random noise added.

---

## 📊 Output Files

### Main Data Files

| File | Description | Rows | Key Columns |
|------|-------------|------|-------------|
| `results_GIQ_SOTA.csv` | Main simulation results | 3,500 | step, trust, system_performance, ai_proportion, transparency |
| `table1_main_results.csv` | Statistical tests | 7 scenarios | mean, sd, p-value, cohen_d, CI |
| `summary_statistics_GIQ.csv` | Descriptive stats | 7 scenarios | mean, median, std, min, max |
| `pareto_frontier_GIQ.csv` | Pareto-optimal configs | ~20 | performance, trust, ai_proportion |

### Supplementary Data

| File | Description | Purpose |
|------|-------------|---------|
| `supp_dense_grid.csv` | Dense AI quota grid (0-60%, Δ=5%) | Optimal quota estimation |
| `supp_do_interventions.csv` | Causal interventions | Robustness to shocks |
| `supp_transparency_factorial.csv` | 2×2×2 factorial design | Transparency decomposition |

### Figures

All figures available in both HTML (interactive) and PNG (publication) formats:

1. **Figure 1**: Performance vs Trust trade-off (scatterplot with convex hull)
2. **Figure 2**: Temporal dynamics (4-panel time series)
3. **Figure 3**: Spatial trajectories (3D party position evolution)
4. **Figure 4**: AI × Transparency interaction (line plot with CI)

---

## 🔒 Reproducibility

### Seeds and Determinism

- All runs use explicit random seeds: `seed + scenario_id * 1000 + replicate`
- Seeds are recorded in output files
- Offline mode is fully deterministic

### Reproduce Published Results

```bash
# Main results
python run_complete_analysis_GIQ.py

# Supplementary analyses
python run_complete_analysis_GIQ.py --supp-dense
python run_complete_analysis_GIQ.py --supp-do
python run_complete_analysis_GIQ.py --supp-factorial
```

### Environment

Use exact versions from `requirements.txt`:

```bash
pip install -r requirements.txt --no-cache-dir
```

**Python version tested**: 3.10.13, 3.11.5, 3.12.1

---

## 📚 Citation

If you use this code or model, please cite:

```bibtex
@article{pablo-marti2025algorithmic,
  title={When Do AI Quotas Improve Legislative Performance? 
         Experimental Evidence from Agent-Based Modeling},
  author={Pablo-Mart{\'i}, Federico and Mir Fern{\'a}ndez, Carlos 
          and Olmeda Martos, Ignacio},
  journal={Government Information Quarterly},
  year={2025},
  note={Under review}
}
```

---

## 👥 Authors

- **Federico Pablo-Martí** - Universidad de Alcalá
- **Carlos Mir Fernández** - Universidad de Alcalá  
- **Ignacio Olmeda Martos** - [Affiliation]

### Correspondence

For questions or collaboration:
- Email: federico.pablo@uah.es
- Repository: [GitHub URL]

---

## 📄 License

This code is provided for **academic and research purposes only**.

- ✅ You may use, modify, and share for non-commercial research
- ✅ Please cite the manuscript when using this code
- ❌ Commercial use requires explicit permission from authors

See `LICENSE` file for full terms.

---

## 🙏 Acknowledgments

This research was supported by [funding information].

Special thanks to:
- Agent-based modeling community
- Reviewers at Government Information Quarterly
- Open-source Python ecosystem (NumPy, Pandas, Plotly, AgentPy)

---

## 📝 Changelog

### Version 1.0 (2025-01-XX) - GIQ Submission
- Rigorous metrics (replaced simulated noise with real calculations)
- Complete statistical analysis pipeline
- Supplementary robustness checks
- Publication-ready figures
- Comprehensive documentation

---

## 🐛 Issues and Contributions

Found a bug? Have a suggestion? Please open an issue on GitHub or contact the authors.

**Note**: This is research code accompanying a manuscript. We may not accept major feature requests until after publication.

---

**Last updated**: October 2025  
**Status**: Under review at Government Information Quarterly
