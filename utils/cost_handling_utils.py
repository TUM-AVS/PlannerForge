import os
import json

# Reads in all cost logs, keeps track of the minima, and returns the sum over these
# This cost function describes the trajectory actually used by the motion planner, since at every time step it chooses the trajectory with the lowest cost for that step
# Note: the returned cost is only good for relative comparisons - currently, not every single time step is logged (this would lead to unnecessarily large storage use)
# Instead, the cost can be used for relative comparisons, when changing a parameter of the motion planner, or an aspect of the scenario
def get_min_cost(folder_path) -> float:
    """
    Calculate the sum of minimum costs from JSON cost logs.
    Returns 0.0 if the folder doesn't exist or contains no valid cost files.
    """
    # Check if the cost directory exists
    if not os.path.exists(folder_path):
        print(f"⚠️  Cost directory not found: {folder_path}")
        print(f"   This usually means the planner failed before generating cost logs.")
        return 0.0
    
    # Check if directory is empty
    if not os.listdir(folder_path):
        print(f"⚠️  Cost directory is empty: {folder_path}")
        return 0.0
    
    min_sum = 0.0
    found_valid_file = False
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    # Extract minimum cost in this file
                    costs = (traj_data["cost"] for traj_data in data.values())
                    min_cost = min(costs, default=None)
                    if min_cost is not None:
                        min_sum += min_cost
                        found_valid_file = True
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"⚠️  Skipping {file_name}: {e}")
    
    if not found_valid_file:
        print(f"⚠️  No valid cost files found in: {folder_path}")
    
    return min_sum

# Create the context that is used by the LLM
def format_analysis(scenario_name: str, cost: float, weights: str, final_output: str, mod_info: str) -> str:
    return f"""\n
=== Simulation Result ===
Scenario: {scenario_name}
Modification Info: {mod_info}
Cost: {cost:.2f}

[Cost Weights]
{weights.strip()}

[Final Output from Motion Planner]
{final_output.strip()}

------------------------
""".strip("\n")
