import os


def load_system_role_for_timestamped_translation():
    role_file = "resources/system_roles/system_role_for_timestamped_translation.txt"
    if not os.path.exists(role_file):
        raise FileNotFoundError(f"System role file '{role_file}' not found.")

    with open(role_file, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_system_role_for_text_correction(mode):
    """
    Load system role for text correction based on the mode (reduce/expand)

    Args:
        mode (str): 'reduce' or 'expand'

    Returns:
        str: System role content
    """
    if mode == "reduce":
        role_file = "resources/system_roles/system_role_for_text_reduction.txt"
    elif mode == "expand":
        role_file = "resources/system_roles/system_role_for_text_expansion.txt"
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes: 'reduce', 'expand'")

    if not os.path.exists(role_file):
        raise FileNotFoundError(f"System role file '{role_file}' not found.")

    with open(role_file, "r", encoding="utf-8") as f:
        return f.read().strip()
