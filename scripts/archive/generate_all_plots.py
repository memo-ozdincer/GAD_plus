#!/usr/bin/env python
"""Generate all publication-quality plots for EXPERIMENTS.tex.

No GPU needed. Reads Parquet data via DuckDB, outputs PNGs.

Usage:
  python scripts/generate_all_plots.py
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import duckdb

from plotting_style import apply_plot_style, palette_color

RUNS = '/lustre07/scratch/memoozd/gadplus/runs'
OUT = '/lustre06/project/6033559/memoozd/GAD_plus/figures'
os.makedirs(OUT, exist_ok=True)

# Consistent style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
})
apply_plot_style()
COLORS = [palette_color(0), palette_color(2), palette_color(1), palette_color(3), palette_color(4), palette_color(5), palette_color(7)]


def fig1_conv_vs_noise():
    """Figure 1: Convergence rate vs noise level (300 samples, main result)."""
    df = duckdb.execute(f"""
        SELECT noise_pm,
               ROUND(100.0 * SUM(CASE WHEN converged THEN 1 ELSE 0 END) / COUNT(*), 1) as rate,
               COUNT(*) as total,
               SUM(CASE WHEN converged THEN 1 ELSE 0 END) as conv
        FROM '{RUNS}/noise_survey_300/summary_*.parquet'
        GROUP BY noise_pm ORDER BY noise_pm
    """).df()

    # Level 0 data points (from pure_gad_sweep, 50 samples each)
    level0_noise = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    level0_rate  = [98, 24, 6, 0, 0, 0, 0, 0, 0, 0, 0]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df['noise_pm'], df['rate'], 'o-', color=COLORS[0], linewidth=2.5,
            markersize=9, label='Level 2: gad_projected (300 samples)', zorder=5)
    ax.plot(level0_noise, level0_rate, 's--', color=COLORS[3], linewidth=2,
            markersize=7, label='Level 0: pure_gad (50 samples)', alpha=0.8, zorder=4)

    ax.axhline(y=70, color=palette_color(7), linestyle=':', alpha=0.5)
    ax.text(150, 72, '70% plateau', color=palette_color(7), fontsize=9)

    ax.set_xlabel('Gaussian Noise on TS Geometry (pm)')
    ax.set_ylabel('Convergence Rate (%)')
    ax.set_title('Noise Robustness: Eckart-Projected GAD vs Unprojected')
    ax.set_ylim(-2, 102)
    ax.set_xlim(-5, 210)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Annotate key points
    ax.annotate('68%', xy=(50, 68), xytext=(60, 80),
                arrowprops=dict(arrowstyle='->', color=COLORS[0]),
                fontsize=10, color=COLORS[0], fontweight='bold')
    ax.annotate('0%', xy=(50, 0), xytext=(60, 12),
                arrowprops=dict(arrowstyle='->', color=COLORS[3]),
                fontsize=10, color=COLORS[3], fontweight='bold')

    fig.savefig(f'{OUT}/fig1_conv_vs_noise.png')
    fig.savefig(f'{OUT}/fig1_conv_vs_noise.pdf')
    plt.close(fig)
    print(f'  Saved fig1_conv_vs_noise')


def fig2_starting_geometry():
    """Figure 2: Convergence by starting geometry (bar chart, 300 samples)."""
    df = duckdb.execute(f"""
        SELECT start_method,
               ROUND(100.0 * SUM(CASE WHEN converged THEN 1 ELSE 0 END) / COUNT(*), 1) as rate,
               SUM(CASE WHEN converged THEN 1 ELSE 0 END)::INT as conv,
               COUNT(*) as total
        FROM '{RUNS}/starting_geom_300/summary_*.parquet'
        GROUP BY start_method ORDER BY rate DESC
    """).df()

    labels = {
        'noised_ts': 'Noised TS\n(10 pm)',
        'midpoint': 'Midpoint\n(R→P)',
        'reactant': 'Reactant',
        'product': 'Product',
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = range(len(df))
    bars = ax.bar(x, df['rate'], color=[COLORS[0], COLORS[1], COLORS[2], COLORS[3]],
                  edgecolor='white', linewidth=1.5, width=0.65)

    for bar, row in zip(bars, df.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{row.rate}%\n({row.conv}/{row.total})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(m, m) for m in df['start_method']], fontsize=12)
    ax.set_ylabel('Convergence Rate (%)')
    ax.set_title('Convergence by Starting Geometry (300 samples)')
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(f'{OUT}/fig2_starting_geometry.png')
    fig.savefig(f'{OUT}/fig2_starting_geometry.pdf')
    plt.close(fig)
    print(f'  Saved fig2_starting_geometry')


def fig3_basin_mapping():
    """Figure 3: Basin stability — RMSD scatter + convergence rate."""
    df = duckdb.execute(f"""
        SELECT noise_pm, rmsd_to_original_ts, same_ts, converged
        FROM '{RUNS}/basin_map/basin_map_results.parquet'
    """).df()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: RMSD scatter
    conv = df[df['converged']]
    same = conv[conv['same_ts']]
    diff = conv[~conv['same_ts']]

    ax1.scatter(same['noise_pm'], same['rmsd_to_original_ts'],
                c=COLORS[0], s=40, alpha=0.6, label='Same TS', zorder=5)
    ax1.scatter(diff['noise_pm'], diff['rmsd_to_original_ts'],
                c=COLORS[3], s=60, alpha=0.8, label='Different TS', marker='x',
                linewidths=2, zorder=6)
    ax1.axhline(0.1, color=palette_color(7), linestyle='--', alpha=0.4, label='Threshold (0.1 Å)')
    ax1.set_xlabel('Noise (pm)')
    ax1.set_ylabel('RMSD to Original TS (Å)')
    ax1.set_title('Basin Stability: Same vs Different TS')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(-0.02, 0.55)

    # Right: Convergence + same-TS rate vs noise
    agg = duckdb.execute(f"""
        SELECT noise_pm,
               COUNT(*) as total,
               SUM(CASE WHEN converged THEN 1 ELSE 0 END)::INT as conv,
               SUM(CASE WHEN same_ts THEN 1 ELSE 0 END)::INT as same
        FROM '{RUNS}/basin_map/basin_map_results.parquet'
        GROUP BY noise_pm ORDER BY noise_pm
    """).df()

    ax2.plot(agg['noise_pm'], 100*agg['conv']/agg['total'], 'o-', color=COLORS[0],
             linewidth=2, markersize=8, label='Converged')
    ax2.plot(agg['noise_pm'], 100*agg['same']/agg['total'], 's--', color=COLORS[1],
             linewidth=2, markersize=7, label='Same TS')
    ax2.fill_between(agg['noise_pm'],
                     100*agg['same']/agg['total'],
                     100*agg['conv']/agg['total'],
                     alpha=0.15, color=COLORS[3], label='Different TS')
    ax2.set_xlabel('Noise (pm)')
    ax2.set_ylabel('Rate (%)')
    ax2.set_title('Convergence and Basin Fidelity')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(-2, 102)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_basin_mapping.png')
    fig.savefig(f'{OUT}/fig3_basin_mapping.pdf')
    plt.close(fig)
    print(f'  Saved fig3_basin_mapping')


def fig4_method_heatmap():
    """Figure 4: Method comparison heatmap (from SLURM logs)."""
    methods = ['gad_small_dt\n(dt=0.005)', 'gad_projected\n(dt=0.01)',
               'gad_tight_clamp', 'gad_adaptive_dt', 'gad_adaptive_tight']
    noise_labels = ['10', '30', '50', '100', '150', '200']
    rates = np.array([
        [96, 96, 84, 66, 48, 20],
        [64, 62, 62, 54, 48, 36],
        [64, 62, 62, 54, 46, 36],
        [54, 34, 30, 20, 16, 6],
        [54, 34, 30, 20, 14, 6],
    ])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(rates, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels([f'{n} pm' for n in noise_labels], fontsize=11)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('Noise Level')
    ax.set_title('Method Comparison: Convergence Rate (%, 50 samples each)')

    for i in range(len(methods)):
        for j in range(len(noise_labels)):
            color = 'white' if rates[i, j] < 40 else 'black'
            ax.text(j, i, f'{rates[i,j]}%', ha='center', va='center',
                    fontweight='bold', fontsize=12, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Convergence Rate (%)')

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig4_method_heatmap.png')
    fig.savefig(f'{OUT}/fig4_method_heatmap.pdf')
    plt.close(fig)
    print(f'  Saved fig4_method_heatmap')


def fig5_trajectory_examples():
    """Figure 5: 2x2 trajectory plots for fast/slow/failure cases."""
    import glob

    # Find representative trajectories from 300-sample noise survey
    # Fast: converged at low step count, low noise
    fast = duckdb.execute(f"""
        SELECT run_id, sample_id, formula, noise_pm, converged_step
        FROM '{RUNS}/noise_survey_300/summary_*.parquet'
        WHERE converged AND noise_pm = 10
        ORDER BY converged_step ASC LIMIT 1
    """).df()

    # Slow: converged at high step count, high noise
    slow = duckdb.execute(f"""
        SELECT run_id, sample_id, formula, noise_pm, converged_step
        FROM '{RUNS}/noise_survey_300/summary_*.parquet'
        WHERE converged AND noise_pm = 100
        ORDER BY converged_step DESC LIMIT 1
    """).df()

    # Failure: did not converge, highest noise
    fail = duckdb.execute(f"""
        SELECT run_id, sample_id, formula, noise_pm, total_steps
        FROM '{RUNS}/noise_survey_300/summary_*.parquet'
        WHERE NOT converged AND noise_pm = 50
        ORDER BY final_force_norm ASC LIMIT 1
    """).df()

    cases = [
        ('Fast Convergence', fast, True),
        ('Slow Convergence', slow, True),
        ('Failure (closest to threshold)', fail, False),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))

    for row_idx, (label, info, is_conv) in enumerate(cases):
        if len(info) == 0:
            continue

        run_id = info['run_id'].iloc[0]
        sample_id = int(info['sample_id'].iloc[0])
        formula = info['formula'].iloc[0]
        noise = int(info['noise_pm'].iloc[0])

        if is_conv:
            conv_step = int(info['converged_step'].iloc[0])
            subtitle = f'{formula}, {noise} pm, conv step {conv_step}'
        else:
            subtitle = f'{formula}, {noise} pm, FAILED'

        traj = duckdb.execute(f"""
            SELECT step, energy, force_norm, n_neg, eig0, eig1,
                   dist_to_known_ts, disp_from_start
            FROM '{RUNS}/noise_survey_300/traj_*.parquet'
            WHERE run_id = '{run_id}' AND sample_id = {sample_id}
            ORDER BY step
        """).df()

        if len(traj) == 0:
            continue

        steps = traj['step'].values

        # Energy
        ax = axes[row_idx, 0]
        ax.plot(steps, traj['energy'], color=COLORS[0], linewidth=1.5)
        ax.set_ylabel('Energy (eV)')
        ax.set_title(f'{label}' if row_idx == 0 else '')
        ax.text(0.02, 0.95, subtitle, transform=ax.transAxes, fontsize=9,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor=palette_color(8), alpha=0.5))
        ax.grid(True, alpha=0.2)

        # Force norm
        ax = axes[row_idx, 1]
        ax.semilogy(steps, traj['force_norm'], color=COLORS[3], linewidth=1.5)
        ax.axhline(0.01, color=COLORS[1], linestyle='--', alpha=0.7, linewidth=1.5,
                   label='threshold')
        ax.set_ylabel('Force Norm (eV/Å)')
        if row_idx == 0:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # n_neg
        ax = axes[row_idx, 2]
        ax.plot(steps, traj['n_neg'], color=palette_color(7), linewidth=1.5,
                drawstyle='steps-post')
        ax.axhline(1, color=COLORS[1], linestyle='--', alpha=0.7, linewidth=1.5,
                   label='target (n_neg=1)')
        ax.set_ylabel('n_neg')
        ax.set_ylim(-0.5, max(traj['n_neg'].max() + 1, 5))
        if row_idx == 0:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # Eigenvalues
        ax = axes[row_idx, 3]
        ax.plot(steps, traj['eig0'], color=COLORS[0], linewidth=1.5, label='λ₁ (lowest)')
        ax.plot(steps, traj['eig1'], color=COLORS[2], linewidth=1.5, label='λ₂')
        ax.axhline(0, color=palette_color(7), linestyle='-', alpha=0.3)
        ax.set_ylabel('Eigenvalue')
        if row_idx == 0:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel('Step')

    fig.suptitle('Representative GAD Trajectories (300-sample noise survey)', fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_trajectories.png')
    fig.savefig(f'{OUT}/fig5_trajectories.pdf')
    plt.close(fig)
    print(f'  Saved fig5_trajectories')


def fig6_steps_vs_noise():
    """Figure 6: Average convergence steps vs noise (shows linear scaling)."""
    df = duckdb.execute(f"""
        SELECT noise_pm,
               ROUND(AVG(CASE WHEN converged THEN converged_step END), 0) as avg_steps,
               ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY CASE WHEN converged THEN converged_step END), 0) as p25,
               ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY CASE WHEN converged THEN converged_step END), 0) as p75
        FROM '{RUNS}/noise_survey_300/summary_*.parquet'
        GROUP BY noise_pm ORDER BY noise_pm
    """).df()

    fig, ax = plt.subplots(figsize=(8, 5))
    mask = df['avg_steps'].notna()
    df_clean = df[mask].copy()
    yerr_lo = np.maximum(df_clean['avg_steps'].values - df_clean['p25'].values, 0)
    yerr_hi = np.maximum(df_clean['p75'].values - df_clean['avg_steps'].values, 0)
    ax.errorbar(df_clean['noise_pm'], df_clean['avg_steps'],
                yerr=[yerr_lo, yerr_hi],
                fmt='o-', color=COLORS[0], linewidth=2, markersize=8,
                capsize=5, capthick=1.5, label='Avg steps (IQR)')

    # Linear fit
    z = np.polyfit(df_clean['noise_pm'], df_clean['avg_steps'], 1)
    x_fit = np.linspace(0, 200, 100)
    ax.plot(x_fit, np.polyval(z, x_fit), '--', color=palette_color(7), alpha=0.5,
            label=f'Linear fit: {z[0]:.1f}·noise + {z[1]:.0f}')

    ax.set_xlabel('Noise (pm)')
    ax.set_ylabel('Steps to Convergence')
    ax.set_title('Convergence Speed vs Noise (300 samples, converged only)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    fig.savefig(f'{OUT}/fig6_steps_vs_noise.png')
    fig.savefig(f'{OUT}/fig6_steps_vs_noise.pdf')
    plt.close(fig)
    print(f'  Saved fig6_steps_vs_noise')


def fig7_irc_rmsd():
    """Figure 7: IRC validation RMSD scatter."""
    df = duckdb.execute(f"""
        SELECT sample_id, formula, intended, half_intended,
               rmsd_reactant, rmsd_product
        FROM '{RUNS}/irc_validation/irc_validation_10pm.parquet'
        WHERE rmsd_reactant IS NOT NULL AND rmsd_product IS NOT NULL
    """).df()

    if len(df) == 0:
        print('  Skipped fig7_irc_rmsd (no data with both RMSDs)')
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    intended = df[df['intended']]
    half = df[df['half_intended']]
    unintended = df[~df['intended'] & ~df['half_intended']]

    ax.scatter(intended['rmsd_reactant'], intended['rmsd_product'],
               c=COLORS[1], s=100, label='Intended', zorder=5, edgecolors='black', linewidth=0.5)
    ax.scatter(half['rmsd_reactant'], half['rmsd_product'],
               c=COLORS[2], s=100, label='Half-intended', zorder=5, edgecolors='black', linewidth=0.5)
    ax.scatter(unintended['rmsd_reactant'], unintended['rmsd_product'],
               c=COLORS[3], s=100, label='Unintended', zorder=5, edgecolors='black', linewidth=0.5)

    ax.axhline(0.3, color=palette_color(7), linestyle='--', alpha=0.3, label='Threshold (0.3 Å)')
    ax.axvline(0.3, color=palette_color(7), linestyle='--', alpha=0.3)

    # Shade intended quadrant
    ax.fill_between([0, 0.3], 0, 0.3, alpha=0.05, color=COLORS[1])

    ax.set_xlabel('RMSD to Reactant (Å)')
    ax.set_ylabel('RMSD to Product (Å)')
    ax.set_title('IRC Validation: Endpoint RMSD (10 pm noise, 10 samples)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, 0.55)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    fig.savefig(f'{OUT}/fig7_irc_rmsd.png')
    fig.savefig(f'{OUT}/fig7_irc_rmsd.pdf')
    plt.close(fig)
    print(f'  Saved fig7_irc_rmsd')


def fig8_avg_force_trajectory():
    """Figure 8: Average force norm over steps for converged vs failed."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, noise_pm, title in [(axes[0], 50, '50 pm noise'), (axes[1], 100, '100 pm noise')]:
        for label, cond, color in [
            ('Converged', 'AND s.converged', COLORS[1]),
            ('Failed', 'AND NOT s.converged', COLORS[3]),
        ]:
            df = duckdb.execute(f"""
                SELECT t.step,
                       AVG(t.force_norm) as avg_force,
                       AVG(CAST(t.n_neg AS DOUBLE)) as avg_nneg
                FROM '{RUNS}/noise_survey_300/traj_*.parquet' t
                JOIN '{RUNS}/noise_survey_300/summary_*.parquet' s
                  ON t.run_id = s.run_id AND t.sample_id = s.sample_id
                WHERE s.noise_pm = {noise_pm} {cond}
                GROUP BY t.step ORDER BY t.step
            """).df()
            if len(df) > 0:
                ax.semilogy(df['step'], df['avg_force'], linewidth=2, color=color,
                           label=label)

        ax.axhline(0.01, color=palette_color(7), linestyle='--', alpha=0.5, label='Threshold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg Force Norm (eV/Å)')
        ax.set_title(f'Average Force Trajectory ({title})')
        ax.legend()
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig8_avg_force_trajectory.png')
    fig.savefig(f'{OUT}/fig8_avg_force_trajectory.pdf')
    plt.close(fig)
    print(f'  Saved fig8_avg_force_trajectory')


def fig9_dist_to_ts():
    """Figure 9: RMSD to known TS over steps (converged runs, multiple noise levels)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, noise_pm in enumerate([10, 50, 100, 200]):
        df = duckdb.execute(f"""
            SELECT t.step, AVG(t.dist_to_known_ts) as avg_dist
            FROM '{RUNS}/noise_survey_300/traj_*.parquet' t
            JOIN '{RUNS}/noise_survey_300/summary_*.parquet' s
              ON t.run_id = s.run_id AND t.sample_id = s.sample_id
            WHERE s.converged AND s.noise_pm = {noise_pm}
            GROUP BY t.step ORDER BY t.step
        """).df()
        if len(df) > 0:
            ax.plot(df['step'], df['avg_dist'], linewidth=2, color=COLORS[i],
                    label=f'{noise_pm} pm')

    ax.set_xlabel('Step')
    ax.set_ylabel('Avg RMSD to Known TS (Å)')
    ax.set_title('Approach to Known TS (converged runs, 300 samples)')
    ax.legend(title='Noise Level')
    ax.grid(True, alpha=0.2)

    fig.savefig(f'{OUT}/fig9_dist_to_ts.png')
    fig.savefig(f'{OUT}/fig9_dist_to_ts.pdf')
    plt.close(fig)
    print(f'  Saved fig9_dist_to_ts')


if __name__ == '__main__':
    print('Generating all plots...')
    print(f'Output: {OUT}/')
    print()

    fig1_conv_vs_noise()
    fig2_starting_geometry()
    fig3_basin_mapping()
    fig4_method_heatmap()
    fig5_trajectory_examples()
    fig6_steps_vs_noise()
    fig7_irc_rmsd()
    fig8_avg_force_trajectory()
    fig9_dist_to_ts()

    print(f'\nAll plots saved to {OUT}/')
