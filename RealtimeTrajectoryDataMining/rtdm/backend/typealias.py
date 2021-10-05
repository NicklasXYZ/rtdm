from typing import List, Tuple, Union

# Define type aliases for
# - Individual tokens that make up sequences
Token = str
# - A list containing tokens
Sequence = List[Token]
# - Patterns that are (frequency, sequence) pairs
Pattern = Tuple[int, Sequence]
# - A 2D coordinate pair
CoordinatePair = Union[Tuple[float, float], List[float]]
