import os


def load_system_role_for_timestamped_translation():
    role_file = "system_role_for_timestamped_translation.txt"
    if not os.path.exists(role_file):
        raise FileNotFoundError(f"System role file '{role_file}' not found.")

    with open(role_file, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_system_role_for_duration_correction(mode):
    if mode == "shorten":
        role_file = "system_role_for_text_shortening.txt"
    elif mode == "extend":
        role_file = "system_role_for_text_extension.txt"
    else:
        raise ValueError(f"Unknown edit mode: {mode}")

    if not os.path.exists(role_file):
        raise FileNotFoundError(f"System role file '{role_file}' not found.")

    with open(role_file, "r", encoding="utf-8") as f:
        return f.read().strip()
