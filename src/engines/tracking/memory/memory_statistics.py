from dataclasses import dataclass

@dataclass
class MemoryStatistics:
    """Profiles global memory metrics of the repository layer."""
    total_memories: int = 0
    average_lifetime_frames: float = 0.0
    longest_lifetime_frames: int = 0
    recovered_memories_count: int = 0
    average_observations: float = 0.0
    average_confidence: float = 0.0
    average_path_length: float = 0.0
    estimated_memory_bytes: int = 0
