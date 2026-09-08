from pathlib import Path
import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#: A run counts as solved when the `result` column says so. Planners that report
#: their verdict in `status` instead (MP-RBFN writes `status=success` and puts a
#: score in `result`) are honoured too — without this fallback their batch plots
#: read 0% success. See docs/integrating_a_planner.md for the column contract.
SUCCESS_LABELS = {"success", "successful"}


def classify_success(df):
    """Boolean Series: True where a scenario was solved.

    Reads `result` first, then falls back to `status` for planners that put a
    numeric score in `result`.
    """
    import pandas as pd
    ok = pd.Series(False, index=df.index)
    if "result" in df.columns:
        ok |= df["result"].astype(str).str.strip().str.lower().isin(SUCCESS_LABELS)
    if "status" in df.columns:
        ok |= df["status"].astype(str).str.strip().str.lower().isin(SUCCESS_LABELS)
    return ok


def success_counts(df):
    """(success, failure, rate_percent) for a score_overview dataframe."""
    ok = classify_success(df)
    n = int(len(ok))
    s = int(ok.sum())
    return s, n - s, (s / n * 100 if n else 0.0)


def make_bar_plot_success_failure():
    frenetix_path = Path(__file__).parent.parent / "Frenetix-Motion-Planner"
    os.makedirs(frenetix_path / "logs", exist_ok=True)
    score_path = frenetix_path / "logs" / "score_overview.csv"

    if not score_path.exists():
        raise Exception("No score file exists.")

    print(f"Reading scores from: {score_path}")

    # Load CSV (semicolon separated)
    df = pd.read_csv(score_path, sep=";")

    # Count scenarios by outcome (see classify_success for the column rule)
    s, f, _ = success_counts(df)
    counts = pd.Series({"Success": s, "Failed": f})

    # Make bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)

    ax.set_title("Scenario Results Overview")
    ax.set_xlabel("Result Type")
    ax.set_ylabel("Number of Scenarios")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    return

def make_bar_plot_failure_analysis():
    frenetix_path = Path(__file__).parent.parent / "Frenetix-Motion-Planner"
    os.makedirs(frenetix_path / "logs", exist_ok=True)
    score_path = frenetix_path / "logs" / "score_overview.csv"

    if not score_path.exists():
        raise Exception("No score file exists.")

    print(f"Reading scores from: {score_path}")

    # Load CSV (semicolon separated)
    df = pd.read_csv(score_path, sep=";")

    # Filter only failures
    df_failures = df[~classify_success(df)]

    # Count by failure message
    counts = df_failures["message"].value_counts()

    # Make bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="tomato")

    ax.set_title("Failure Analysis by Message")
    ax.set_xlabel("Failure Reason (message)")
    ax.set_ylabel("Number of Scenarios")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    plt.show()

import yaml
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

def make_combined_bar_plot(logs_path: Path, weights_path: Path):
    score_path = logs_path / "score_overview.csv"

    if not score_path.exists():
        raise Exception("No score file exists.")

    print(f"Reading scores from: {score_path}")

    # Load CSV (semicolon separated)
    df = pd.read_csv(score_path, sep=";")
    
    # Identify which preset mode is being used
    cost_mode = compare_to_templates(weights_path) if weights_path.exists() else "Unknown"

    # Define consistent color palette
    COLOR_SUCCESS = '#2ECC71'  # Green
    COLOR_FAILURE = '#E74C3C'  # Red (general failure)
    COLOR_COLLISION = '#C0392B'  # Dark Red (collision)
    COLOR_LINE_GRAY = '#2C3E50'  # Line gray
    COLOR_GRID = '#ECF0F1'  # Light gray
    
    # Define failure-reason severity colors
    FAILURE_COLORS = {
        'collision': '#C0392B',  # Critical - Dark Red
        'no valid or feasible trajectory found': '#E74C3C',  # High - Red
        'time limit reached': '#F39C12',  # Medium - Orange
        'timeout reached': '#F39C12',  # Medium - Orange (alternative name)
        'goal reached faster': '#3498DB',  # Low - Blue
        'goal reached out of time': '#5DADE2',  # Low - Light Blue
    }
    
    # Unified font size system
    FONT_TITLE = 18         # Main title (increased from 14)
    FONT_SUBPLOT = 16        # Subplot titles (increased from 12)
    FONT_LABEL = 12          # Labels, legend (increased from 10)
    FONT_TICK = 14           # Tick labels (increased from 9)
    FONT_VALUE = 14          # Value labels (increased from 10)
    
    def get_failure_color(failure_msg):
        """Get color based on failure message severity"""
        failure_lower = str(failure_msg).lower()
        for key, color in FAILURE_COLORS.items():
            if key in failure_lower:
                return color
        return COLOR_FAILURE  # Default to red if not recognized
    
    # --- Create figure with 3 vertical plots ---
    fig = plt.figure(figsize=(16, 13), facecolor='white')
    
    # Use GridSpec for 3 rows (plots 1 & 2 bigger, plot 3 smaller)
    gs = fig.add_gridspec(3, 1, left=0.18, right=0.95, top=0.94, bottom=0.05, 
                         hspace=0.35, height_ratios=[0.28, 0.42, 0.30])
    
    # Center title over the plot area (not the full figure)
    title_x = (0.18 + 0.95) / 2  # Midpoint between left and right margins
    fig.suptitle('Batch Simulation Analysis', fontsize=FONT_TITLE, fontweight='bold', 
                 x=title_x, y=0.97)
    
    ax1 = fig.add_subplot(gs[0])  # Top - Cost Weights
    ax2 = fig.add_subplot(gs[1])  # Middle - Donut Chart
    ax3 = fig.add_subplot(gs[2])  # Bottom - Failure Bar Chart

    # --- Plot 1: Cost weights table (compact) ---
    ax1.axis("off")
    if weights_path.exists():
        with open(weights_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        cost_weights = yaml_data.get("cost_weights", {})
        if cost_weights:
            # Format labels and values (shorter labels)
            col_labels = [label.replace("_", "\n") for label in cost_weights.keys()]
            row_values = [[f'{v:.1f}' for v in cost_weights.values()]]

            # Create compact table
            table = ax1.table(cellText=row_values,
                            colLabels=col_labels,
                            loc="center",
                            cellLoc="center",
                            colLoc="center")
            
            table.auto_set_font_size(False)
            table.set_fontsize(FONT_TICK)
            table.scale(1, 2.5)
            
            # Style the table
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_facecolor(COLOR_LINE_GRAY)
                    cell.set_text_props(weight='bold', color='white', fontsize=FONT_TICK)
                else:  # Data rows
                    cell.set_facecolor('#f8f9fa')
                    cell.set_text_props(fontsize=FONT_VALUE, weight='bold')
                cell.set_edgecolor(COLOR_LINE_GRAY)
                cell.set_linewidth(1.5)
            
            # Add title for table with mode name
            ax1.text(0.5, 0.95, f'Cost Weight Configuration (Mode: {cost_mode})', 
                    transform=ax1.transAxes, fontsize=FONT_SUBPLOT, 
                    fontweight='bold', ha='center', va='top')

    # --- Plot 2: Success / Failure overview (Donut Chart) ---
    if {"result", "status"} & set(df.columns) and len(df):
        success_count, failure_count, success_rate = success_counts(df)
        counts_overview = pd.Series({"Success": success_count,
                                     "Failed": failure_count})
        counts_overview = counts_overview[counts_overview > 0]

        colors = [COLOR_SUCCESS if 'success' in str(label).lower() else COLOR_FAILURE
                  for label in counts_overview.index]
        total = int(counts_overview.sum())
        
        # Create donut chart (cleaner - no labels on wedges, use legend instead)
        wedges, _ = ax2.pie(counts_overview.values, 
                           labels=None,  # Remove labels on the ring for cleaner look
                           colors=colors,
                           startangle=90,
                           wedgeprops=dict(width=0.4, edgecolor='black', linewidth=2))
        
        # Add legend below the donut
        ax2.legend(wedges,
                  [f"{lab}: {val}" for lab, val in zip(counts_overview.index, counts_overview.values)],
                  loc="lower center", bbox_to_anchor=(0.5, -0.1),
                  ncol=2, frameon=False, fontsize=FONT_LABEL)
        
        # Add success rate in the center
        ax2.text(0, 0, f'{success_rate:.1f}%\nSuccess Rate', 
                ha='center', va='center',
                fontsize=FONT_SUBPLOT + 4, fontweight='bold', 
                color=COLOR_SUCCESS if success_rate >= 50 else COLOR_FAILURE)
        
        ax2.set_title("Scenario Results", fontsize=FONT_SUBPLOT, fontweight='bold', pad=10)
    else:
        ax2.set_title("Scenario Results (no data)", fontsize=FONT_SUBPLOT, fontweight='bold')
        ax2.text(0.5, 0.5, "No results available", ha="center", va="center", 
                fontsize=FONT_LABEL, color='gray')
        ax2.axis("off")

    # --- Plot 3: Failure analysis (Horizontal Bar Chart with percentages) ---
    if "result" in df.columns and "message" in df.columns:
        # Define what counts as "success"
        success_labels = {"Success", "success", "successful"}
        df_failures = df[~df["result"].isin(success_labels)]

        if not df_failures.empty:
            counts_failures = df_failures["message"].value_counts()
            if not counts_failures.empty:
                # Sort by count (ascending for horizontal bar chart)
                counts_failures = counts_failures.sort_values(ascending=True)
                
                # Calculate total failures for percentages
                total_failures = counts_failures.sum()
                
                # Assign severity-based colors
                colors = [get_failure_color(reason) for reason in counts_failures.index]
                
                # Horizontal bar chart with shorter bars
                bars = ax3.barh(range(len(counts_failures)), counts_failures.values,
                               height=0.7, color=colors, edgecolor='black', linewidth=1.5, alpha=0.9)
                ax3.set_yticks(range(len(counts_failures)))
                # Larger font size for better readability
                ax3.set_yticklabels(counts_failures.index, fontsize=FONT_LABEL, fontweight='normal')
                
                # Add value labels with percentages to the right of bars
                max_value = counts_failures.max()
                for i, (bar, value) in enumerate(zip(bars, counts_failures.values)):
                    width = bar.get_width()
                    percentage = (value / total_failures * 100)
                    # Place text inside bar, on the right side
                    ax3.text(width - max_value * 0.01, bar.get_y() + bar.get_height()/2.,
                            f'{int(value)} ({percentage:.1f}%)',
                            ha='right', va='center', fontsize=FONT_VALUE, fontweight='bold', color='white')
                
                ax3.set_title("Failure Reasons", fontsize=FONT_SUBPLOT, fontweight='bold', pad=10)
                ax3.set_xlabel("Count", fontsize=FONT_LABEL, fontweight='bold')
                ax3.set_ylabel("")  # Remove y-label to save space
                ax3.tick_params(axis='both', labelsize=FONT_TICK)
                ax3.tick_params(axis='y', pad=2)  # Reduce y-tick label padding for alignment
                ax3.grid(axis="x", linestyle="--", alpha=0.3, color=COLOR_GRID, linewidth=0.8)
                ax3.set_axisbelow(True)
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)
                ax3.spines['left'].set_color(COLOR_LINE_GRAY)
                ax3.spines['bottom'].set_color(COLOR_LINE_GRAY)
            else:
                ax3.set_title("Failure Reasons", fontsize=FONT_SUBPLOT, fontweight='bold')
                ax3.text(0.5, 0.5, "✓ All scenarios successful!", ha="center", va="center",
                        fontsize=FONT_LABEL, color=COLOR_SUCCESS, fontweight='bold')
                ax3.axis("off")
        else:
            ax3.set_title("Failure Reasons", fontsize=FONT_SUBPLOT, fontweight='bold')
            ax3.text(0.5, 0.5, "✓ All scenarios successful!", ha="center", va="center",
                    fontsize=FONT_LABEL, color=COLOR_SUCCESS, fontweight='bold')
            ax3.axis("off")
    else:
        ax3.set_title("Failure Reasons", fontsize=FONT_SUBPLOT, fontweight='bold')
        ax3.text(0.5, 0.5, "Required columns missing", ha="center", va="center",
                fontsize=FONT_LABEL, color='gray')
        ax3.axis("off")

    # --- Align ax2 and ax3 to the same box width/center as ax1 (table) ---
    # This ensures all three plots share the same left and right boundaries
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    
    # Plot 2 (donut): match plot 1's full width
    ax2.set_position([pos1.x0, pos2.y0, pos1.width, pos2.height])
    ax2.set_aspect("equal", adjustable="box")  # Keep donut circular
    
    # Plot 3 (bar): 9/13 of plot 1's width, centered
    new_width = pos1.width * (9/13)
    x_offset = (pos1.width - new_width) / 2
    ax3.set_position([pos1.x0 + x_offset, pos3.y0, new_width, pos3.height])
    
    # Don't use tight_layout since we're using GridSpec for precise alignment
    plt.savefig(logs_path / "batch_stats.png", format="png", dpi=150)
    # plt.show()
    return plt

# Note: For visualization, distance_to_reference_path and distance_to_obstacles are normalized
# Original values: distance_to_reference_path (1.0-4.0) → normalized by /4.0
#                 distance_to_obstacles (0.0-2.0) → normalized by /2.0
PRESETS = {
    "Default": {
        "acceleration": 0.0, "jerk": 0.0, "lateral_jerk": 0.2, "longitudinal_jerk": 0.2,
        "orientation_offset": 0.0, "path_length": 0.0, "lane_center_offset": 0.0,
        "velocity_offset": 1.0, "velocity": 0.0, "distance_to_reference_path": 0.75,  # 3.0/4.0
        "distance_to_obstacles": 0.0, "prediction": 0.2, "responsibility": 0.0,
    },
    "Comfort": {
        "acceleration": 0.2, "jerk": 0.2, "lateral_jerk": 0.8, "longitudinal_jerk": 0.8,
        "orientation_offset": 0.0, "path_length": 0.0, "lane_center_offset": 0.5,
        "velocity_offset": 0.3, "velocity": 0.0, "distance_to_reference_path": 0.25,  # 1.0/4.0
        "distance_to_obstacles": 0.15, "prediction": 0.2, "responsibility": 0.1,  # 0.3/2.0
    },
    "Balanced": {
        "acceleration": 0.3, "jerk": 0.3, "lateral_jerk": 0.5, "longitudinal_jerk": 0.5,
        "orientation_offset": 0.1, "path_length": 0.2, "lane_center_offset": 0.7,
        "velocity_offset": 0.6, "velocity": 0.2, "distance_to_reference_path": 0.5,  # 2.0/4.0
        "distance_to_obstacles": 0.4, "prediction": 0.3, "responsibility": 0.2,  # 0.8/2.0
    },
    "Efficiency-Sporty": {
        "acceleration": 0.2, "jerk": 0.15, "lateral_jerk": 0.25, "longitudinal_jerk": 0.25,
        "orientation_offset": 0.2, "path_length": 0.6, "lane_center_offset": 0.6,
        "velocity_offset": 1.0, "velocity": 0.4, "distance_to_reference_path": 0.5,  # 2.0/4.0
        "distance_to_obstacles": 0.3, "prediction": 0.2, "responsibility": 0.1,  # 0.6/2.0
    },
    "Safety-Conservative": {
        "acceleration": 0.0, "jerk": 0.0, "lateral_jerk": 0.2, "longitudinal_jerk": 0.2,
        "orientation_offset": 0.0, "path_length": 0.0, "lane_center_offset": 0.0,
        "velocity_offset": 1.0, "velocity": 0.0, "distance_to_reference_path": 5.0,
        "distance_to_obstacles": 1.0, "prediction": 0.5, "responsibility": 0.3,
    }
}


def plot_driving_modes_comparison():
    """
    Create a line plot comparing all 5 driving mode presets.
    Returns a matplotlib figure.
    """
    # Heatmap of cost-weight profiles: rows = driving modes, cols = parameters.
    # (A line chart connects categorical parameters and gets dominated by the one
    #  large distance_to_reference_path weight; a heatmap shows every value clearly.)
    INK = "#2b2d42"
    modes = list(PRESETS.keys())
    param_names = list(PRESETS["Default"].keys())
    labels = [name.replace("_", " ").title() for name in param_names]
    M = np.array([[PRESETS[m][p] for p in param_names] for m in modes], dtype=float)

    with plt.rc_context({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "axes.titlecolor": INK}):
        fig, ax = plt.subplots(figsize=(14, 4.8), facecolor="white")
        im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(M.max(), 1e-9))
        ax.set_xticks(range(len(param_names)))
        ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=9)
        ax.set_yticks(range(len(modes)))
        ax.set_yticklabels(modes, fontsize=10.5)
        # white separators between cells
        ax.set_xticks(np.arange(-.5, len(param_names), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(modes), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(len(modes)):
            for j in range(len(param_names)):
                v = M[i, j]
                ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > M.max() * 0.45 else INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8)
        ax.set_title("Cost-weight profiles by driving mode",
                     fontsize=14, fontweight="semibold", loc="left", pad=10)
        fig.tight_layout()
    return fig

def compare_to_templates(weights_path: Path) -> str:
    """
    Compare a cost weights YAML file against predefined presets.
    Returns the preset name if exact match is found, otherwise 'Custom'.
    """

    # Load YAML
    with open(weights_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Extract cost_weights section
    cost_weights = yaml_data.get("cost_weights", {})
    if not cost_weights:
        return "Custom"

    # Normalize to floats for safe comparison
    normalized = {k: float(v) for k, v in cost_weights.items() if v is not None}

    # Compare with each preset
    for preset_name, preset_values in PRESETS.items():
        # Must have same keys and same values (exact match)
        if set(normalized.keys()) == set(preset_values.keys()):
            all_match = all(
                float(normalized[k]) == float(preset_values[k])
                for k in preset_values.keys()
            )
            if all_match:
                return preset_name

    # No exact match found
    return "Custom"

def make_batch_summary_string(logs_path: Path, weights_path: Path, batch_type: str):
    score_path = logs_path / "score_overview.csv"

    if not score_path.exists():
        raise FileNotFoundError(f"No score file exists at {score_path}")

    # Determine cost model
    if not weights_path.exists():
        cost_model = "Not Supported"
    else:
        cost_model = compare_to_templates(weights_path=weights_path)

    print(f"Reading scores from: {score_path}")
    df = pd.read_csv(score_path, sep=";")

    # --- Success counting ---
    is_success = classify_success(df)
    total = len(df)
    successes = int(is_success.sum())
    failures = total - successes

    # --- Average timestep analysis ---
    avg_time_all = df["timestep"].mean()
    avg_time_success = df.loc[is_success, "timestep"].mean() if successes > 0 else None
    avg_time_failure = df.loc[~is_success, "timestep"].mean() if failures > 0 else None

    avg_time_str = (
        f"Average time until outcome:\n"
        f"    - Overall: {avg_time_all:.1f} timesteps\n"
    )
    if avg_time_success is not None:
        avg_time_str += f"    - Success: {avg_time_success:.1f} timesteps\n"
    if avg_time_failure is not None:
        avg_time_str += f"    - Failure: {avg_time_failure:.1f} timesteps\n"

    # --- Failure analysis ---
    failure_lines = []
    if failures > 0 and "message" in df.columns:
        df_failures = df.loc[~is_success]
        message_counts = df_failures["message"].value_counts()
        for msg, count in message_counts.items():
            failure_lines.append(f"    - {msg}: {count} out of {failures}")
    else:
        failure_lines.append("    - No failures")

    # --- Final summary string ---
    summary = (
        f"---------\nBatch Simulation\n"
        f"Batch Selection: {batch_type}\n"
        f"Cost Model: {cost_model}\n"
        f"Success: {successes} out of {total}\n"
        f"{avg_time_str}"
        f"Failure Analysis (Messages):\n" +
        "\n".join(failure_lines)
    )

    return summary


def make_performance_plot(logs_path: Path, weights_path: Path):
    """Compare the current (active) cost model against all 5 preset driving
    modes as a heatmap of the 13 cost weights. The current model is added as a
    6th row and highlighted, so you can see exactly how it differs from
    Default / Comfort / Balanced / Efficiency-Sporty / Safety-Conservative."""
    import yaml
    from matplotlib.patches import Rectangle
    INK = "#2b2d42"; ACCENT = "#2a9d8f"

    def _placeholder(msg, color="gray"):
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, color=color)
        ax.axis("off")
        return fig

    if not weights_path.exists() or str(weights_path) == "non_existent":
        return _placeholder("No cost configuration available")
    try:
        cost_config = yaml.safe_load(open(weights_path)) or {}
    except Exception as e:
        return _placeholder(f"Error loading config: {e}", "red")
    cost_weights = cost_config.get("cost_weights", {})
    if not cost_weights:
        return _placeholder("No cost weights found in configuration")

    driving_mode = compare_to_templates(weights_path)          # matched preset or "Custom"
    current = {k: v for k, v in cost_weights.items() if k != "external_cost_weights"}

    params = list(PRESETS["Default"].keys())                   # 13 ordered cost-weight keys
    preset_names = list(PRESETS.keys())                        # the 5 presets
    rows = preset_names + [f"Current ({driving_mode})"]
    M = np.array(
        [[float(PRESETS[m][p]) for p in params] for m in preset_names]
        + [[float(current.get(p, 0.0)) for p in params]],
        dtype=float)
    labels = [p.replace("_", " ").title() for p in params]

    with plt.rc_context({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "axes.titlecolor": INK}):
        fig, ax = plt.subplots(figsize=(14, 5.6), facecolor="white")
        vmax = max(M.max(), 1e-9)
        im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=9)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=10.5)
        ax.set_xticks(np.arange(-.5, len(params), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(len(rows)):
            for j in range(len(params)):
                v = M[i, j]
                ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > vmax * 0.45 else INK)
        cur_i = len(rows) - 1
        ax.get_yticklabels()[cur_i].set_color(ACCENT)
        ax.get_yticklabels()[cur_i].set_fontweight("bold")
        ax.add_patch(Rectangle((-.5, cur_i - .5), len(params), 1, fill=False,
                               edgecolor=ACCENT, linewidth=2.5, zorder=5))
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8)
        ax.set_title("Driving-mode cost weights: 5 presets + current (highlighted)",
                     fontsize=14, fontweight="semibold", loc="left", pad=10)
        fig.tight_layout()
    return fig


def make_success_rate_plot(logs_path: Path):
    """Create detailed success/failure analysis with timestep distribution."""
    score_path = logs_path / "score_overview.csv"
    if not score_path.exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No batch data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    df = pd.read_csv(score_path, sep=";")

    # --- clean, modern palette + styling ---
    SUCCESS = "#2a9d8f"   # muted teal-green
    FAILURE = "#e76f51"   # soft terracotta
    INK     = "#2b2d42"   # dark slate (text)
    SUBTLE  = "#8d99ae"   # muted grey
    GRID    = "#e7eaee"
    success_labels = {"Success", "success", "successful"}
    is_success = df["result"].isin(success_labels)

    def _clean(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9ced6")
        ax.tick_params(colors=INK, labelsize=10, length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_facecolor("white")

    rc = {
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.labelsize": 11, "axes.titlesize": 12.5, "figure.dpi": 120,
    }
    with plt.rc_context(rc):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 9.5), facecolor="white")

        # Plot 1: Overall success / failure
        success_count = int(is_success.sum())
        failure_count = int((~is_success).sum())
        total_count = len(df)
        success_rate = (success_count / total_count * 100) if total_count else 0
        bars = ax1.bar(["Success", "Failure"], [success_count, failure_count],
                       color=[SUCCESS, FAILURE], width=0.62, zorder=3)
        _clean(ax1)
        ax1.set_ylabel("Scenarios")
        ax1.set_title(f"Outcome  ·  {success_rate:.1f}% success", fontweight="semibold", loc="left", pad=8)
        ax1.margins(y=0.18)
        for bar, c in zip(bars, [success_count, failure_count]):
            pct = (c / total_count * 100) if total_count else 0
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{c}\n{pct:.0f}%", ha="center", va="bottom", fontsize=11, color=INK, linespacing=1.3)

        # Plot 2: Timestep distribution (overlaid, soft fills)
        ts = df["timestep"].dropna()
        if len(ts) and ts.max() > ts.min():
            bins = np.linspace(ts.min(), ts.max(), 16)
            ax2.hist(df.loc[is_success, "timestep"], bins=bins, color=SUCCESS, alpha=0.55,
                     label="Success", edgecolor="white", linewidth=0.6, zorder=3)
            ax2.hist(df.loc[~is_success, "timestep"], bins=bins, color=FAILURE, alpha=0.55,
                     label="Failure", edgecolor="white", linewidth=0.6, zorder=3)
            leg = ax2.legend(frameon=False, fontsize=10)
        _clean(ax2)
        ax2.set_xlabel("Timestep at outcome")
        ax2.set_ylabel("Scenarios")
        ax2.set_title("When outcomes occur", fontweight="semibold", loc="left", pad=8)

        # Plot 3: Success rate by timestep range (single accent + 50% reference)
        df_sorted = df.sort_values("timestep").copy()
        if df_sorted["timestep"].notna().sum() and df_sorted["timestep"].nunique() > 1:
            df_sorted["bin"] = pd.cut(df_sorted["timestep"], bins=8)
            df_sorted["_ok"] = df_sorted["result"].isin(success_labels)
            rate = df_sorted.groupby("bin", observed=True)["_ok"].mean() * 100
            labels = [f"{int(i.left)}–{int(i.right)}" for i in rate.index]
            xs = range(len(rate))
            ax3.axhline(50, color=SUBTLE, linestyle=(0, (4, 4)), linewidth=1, zorder=1)
            b3 = ax3.bar(xs, rate.values, color=SUCCESS, width=0.7, zorder=3)
            for x, v in zip(xs, rate.values):
                ax3.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
            ax3.set_xticks(list(xs))
            ax3.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
            ax3.set_ylim(0, 105)
        _clean(ax3)
        ax3.set_ylabel("Success rate (%)")
        ax3.set_xlabel("Timestep range")
        ax3.set_title("Success rate vs. duration", fontweight="semibold", loc="left", pad=8)

        # Plot 4: Failure reasons — horizontal bars (sorted), instead of a pie
        if failure_count > 0 and "message" in df.columns:
            fc = df.loc[~is_success, "message"].astype(str).str.strip().value_counts().head(8)
            order = fc.iloc[::-1]                      # largest on top
            ypos = range(len(order))
            shades = plt.cm.OrRd(np.linspace(0.45, 0.85, len(order)))
            ax4.barh(list(ypos), order.values, color=shades, zorder=3, height=0.7)
            ax4.set_yticks(list(ypos))
            ax4.set_yticklabels([t if len(t) <= 34 else t[:31] + "…" for t in order.index], fontsize=9)
            for y, v in zip(ypos, order.values):
                ax4.text(v + max(order.values) * 0.01, y, str(int(v)), va="center", fontsize=9, color=INK)
            for s in ("top", "right"):
                ax4.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax4.spines[s].set_color("#c9ced6")
            ax4.tick_params(colors=INK, length=0)
            ax4.set_axisbelow(True)
            ax4.grid(axis="x", color=GRID, linewidth=1)
            ax4.margins(x=0.12)
            ax4.set_xlabel("Scenarios")
        else:
            ax4.text(0.5, 0.5, "No failures", ha="center", va="center",
                     fontsize=15, color=SUCCESS, fontweight="semibold")
            ax4.axis("off")
        ax4.set_title("Failure reasons", fontweight="semibold", loc="left", pad=8)

        fig.suptitle(f"Batch Simulation Analysis  ·  {total_count} scenarios",
                     fontsize=15, fontweight="semibold", color=INK, x=0.012, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def make_cost_analysis_plot(logs_path: Path, weights_path: Path):
    """Create cost-related analysis plots."""
    score_path = logs_path / "score_overview.csv"
    if not score_path.exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No batch data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    df = pd.read_csv(score_path, sep=";")
    cost_mode = compare_to_templates(weights_path) if weights_path.exists() else "Unknown"

    SUCCESS = "#2a9d8f"; FAILURE = "#e76f51"; INK = "#2b2d42"; GRID = "#e7eaee"
    success_labels = {"Success", "success", "successful"}
    d = df.sort_values("timestep", ascending=False).head(20)
    ok = d["result"].astype(str).str.strip().isin(success_labels)

    with plt.rc_context({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "axes.titlecolor": INK}):
        fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
        y_pos = np.arange(len(d))
        bars = ax.barh(y_pos, d["timestep"].values,
                       color=[SUCCESS if o else FAILURE for o in ok], height=0.72, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(d["scenario"].values, fontsize=8.5)
        ax.invert_yaxis()
        mx = d["timestep"].max() if len(d) else 1
        for b, v in zip(bars, d["timestep"].values):
            ax.text(v + mx * 0.01, b.get_y() + b.get_height() / 2, f"{int(v)}",
                    va="center", fontsize=8, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9ced6")
        ax.tick_params(length=0, colors=INK)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.margins(x=0.06)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=SUCCESS, label="Success"),
                           Patch(color=FAILURE, label="Failure")],
                  frameon=False, loc="lower right", fontsize=10)
        ax.set_xlabel("Timestep (duration)")
        ax.set_title(f"Longest 20 scenarios  ·  cost model: {cost_mode}",
                     fontsize=13.5, fontweight="semibold", loc="left", pad=10)
        fig.tight_layout()
    return fig


def make_batch_dataframe(logs_path: Path) -> pd.DataFrame:
    """Load batch simulation results as DataFrame."""
    score_path = logs_path / "score_overview.csv"
    if not score_path.exists():
        return pd.DataFrame({"Message": ["No batch data available"]})

    df = pd.read_csv(score_path, sep=";")
    display_df = df[["scenario", "result", "timestep", "message"]].copy()
    display_df.columns = ["Scenario", "Result", "Timestep", "Details"]
    return display_df


# make_combined_bar_plot(logs_path=Path("<repo>/RBFN-Motion-Primitives/logs/score_overview.csv"), weights_path=Path("").resolve())
# make_combined_bar_plot(logs_path=Path("<repo>/RBFN-Motion-Primitives/logs").resolve(), weights_path=Path("none"))
# print(make_batch_summary_string(Path("<repo>/Frenetix-Motion-Planner/logs"), Path("<repo>/Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml"), "Base 100 Scenarios"))