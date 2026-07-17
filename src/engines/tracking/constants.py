# Centralized Tracking Parameters Constants
DEFAULT_HISTORY_SIZE = 30
DEFAULT_MAX_LOST_FRAMES = 30
DEFAULT_TENTATIVE_FRAMES = 3
DEFAULT_MIN_CONFIDENCE = 0.50

# State Label string mappings for reports and visual indicators
STATE_LABELS = {
    1: "Tentative",
    2: "Confirmed",
    3: "Tracked",
    4: "Temporarily Lost",
    5: "Recovered",
    6: "Exited",
    7: "Removed"
}
