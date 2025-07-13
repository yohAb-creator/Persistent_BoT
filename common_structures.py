from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ThoughtNode:
    """A shared class for the ThoughtNode"""
    id: int
    text: str
    depth: int
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    score: float = 0.0
    paradigm: Optional[str] = None
    is_promising: bool = True

    def __hash__(self): return hash(self.id)
    def __eq__(self, other):
        if not isinstance(other, ThoughtNode): return NotImplemented
        return self.id == other.id