"""Dictionary utilities."""
from typing import Any, Dict, List

Example = Dict[str, Any]


def shuffle_dict(d: dict, seed=61723) -> dict:
    """Shuffle dict keys.
    
    Starting from Python 3.7 dictionaries preserve insertion order.
    This method randomly shuffles the elements in the dict.
    """
    import random 
    random.seed(seed)
    # Create a tuple w/ the key-value pairs
    d = list(d.items())

    # Shuffle the list
    random.shuffle(d)
    return dict(d)


def update_examples(examples: List[Example], key: str, values: List[Any]):
    """Adds the specified values to the examples under the specified key.
    
    Syntactic sugar method to avoid boilerplate code. It is a
    non-idempotent method, modifying the examples in place.
    """
    assert len(examples) == len(values)

    for example, value in zip(examples, values):
        example[key] = value