import json
import re
import random
import yaml

def generate_modified_scenario_name(scenario_pre: str, mod_type: str) -> str:
    """
    Generates a new scenario name by updating or appending a modification suffix.

    Rules:
        - Existing suffix like _T1234, _TB5678, etc. is detected and replaced.
        - The new suffix includes a sorted combination of modifier types (T, B, P).
        - The modifier types are sorted in the order: T, B, P.
        - A new random 4-digit number is appended as the unique identifier.

    Parameters:
        scenario_pre (str): Existing scenario name (may already include a mod suffix).
        mod_type (str): One or more modifier type letters (T, B, P).

    Returns:
        str: Updated scenario name with new suffix.
    """
    MODIFIER_ORDER = ['T', 'B', 'P']
    random_id = random.randint(1000, 9999)  # always 4-digit

    # Match existing suffix pattern like _TBP1234 at the end
    match = re.search(r'_(?P<modifiers>[TBP]{1,3})(?P<number>\d{1,5})$', scenario_pre)

    if match:
        existing_mods = set(match.group("modifiers"))
        new_mods = set(mod_type)
        combined_mods = existing_mods.union(new_mods)
        ordered_mods = ''.join([m for m in MODIFIER_ORDER if m in combined_mods])
        base_name = scenario_pre[:match.start()]
    else:
        ordered_mods = ''.join([m for m in MODIFIER_ORDER if m in set(mod_type)])
        base_name = scenario_pre

    return f"{base_name}_{ordered_mods}{random_id}"

def check_yaml_structure(original_yaml: str, modified_yaml: str) -> bool:
    """
    Checks if the modified YAML has the same structure (keys & nesting) as the original YAML,
    ignoring commented lines.
    """
    def load_structure(yaml_text: str):
        # Remove commented lines before parsing
        uncommented = "\n".join(
            line for line in yaml_text.splitlines()
            if not line.strip().startswith("#") and line.strip() != ""
        )
        try:
            parsed = yaml.safe_load(uncommented)
        except yaml.YAMLError:
            return None
        return parsed

    original_data = load_structure(original_yaml)
    modified_data = load_structure(modified_yaml)

    if original_data is None or modified_data is None:
        return False

    # Recursively check keys
    def compare_keys(d1, d2):
        if isinstance(d1, dict) and isinstance(d2, dict):
            return set(d1.keys()) == set(d2.keys()) and all(
                compare_keys(d1[k], d2[k]) for k in d1
            )
        elif isinstance(d1, list) and isinstance(d2, list):
            return all(compare_keys(i1, i2) for i1, i2 in zip(d1, d2))
        else:
            # Values are ignored — only structure matters
            return True

    return compare_keys(original_data, modified_data)

def extract_clean_yaml(output_str: str) -> str:
    """
    Extracts only the YAML configuration from an LLM output by
    finding the first 'cost_weights:' and the last valid parameter line.
    Strips out any extra Markdown formatting like ```yaml or ``` at the end.
    """
    lines = output_str.splitlines()

    # Remove common code block markers
    lines = [line for line in lines if not line.strip().startswith("```")]

    # Identify the first line with 'cost_weights:'
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("cost_weights:"):
            start_idx = i
            break

    if start_idx is None:
        return None

    # Identify the last parameter line — here, occ_ve is the last real key
    end_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if line := lines[i].strip():
            if line.startswith("occ_ve:") or line.startswith("occ_ve: "):
                end_idx = i
                break

    if end_idx is None:
        return None

    # Slice and join back
    clean_yaml = "\n".join(lines[start_idx:end_idx + 1])

    return clean_yaml

def build_batch_options(loc: dict, loc_ids: list, tags: list, tags_ids: list, road: dict, road_ids: list, obs: dict, obs_ids: list, vel: dict, vel_ids: list) -> str:
    full_string = f"""### 1. Location: 
    - {loc}
    - Candidates: {len(loc_ids)}
    ### 2. Tags:
    - {tags}
    - Candidates: {len(tags_ids)}
    ### 3. Road Network:
    - {road}
    - Candidates: {len(road_ids)}
    ### 4. Obstacles:
    - {obs}
    - Candidates: {len(obs_ids)}
    ### 5. Velocity:
    - {vel}
    - Candidates: {len(vel_ids)}"""
    return full_string

def separate_text_and_leading_json(input: str) -> tuple[dict, str]:
    """
    Separates a leading JSON object (optionally wrapped in Markdown/format markers)
    from the rest of the string.

    Args:
        input (str): Text that may start with a JSON object followed by other text.

    Returns:
        tuple: (parsed_json: dict | None, remaining_text: str)
    """
    text = input.strip()

    # Remove common format markers (```json, ``` , '''json, ''')
    text = re.sub(r"^(```json|```|'''json|''')\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\n```|\n''')\s*$", "", text)

    # Look for the first JSON object at the beginning
    match = re.match(r'^\s*(\{.*?\})', text, re.DOTALL)
    if not match:
        return None, text

    json_str = match.group(1)
    remaining = text[match.end():].strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # If parsing fails, treat everything as plain text
        return None, input.strip()

    return parsed, remaining

