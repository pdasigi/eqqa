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


def unfold_to_list(dict_of_dicts: Dict[str, Dict], col1, col2) -> List[Dict[str, Any]]:
    """Unfolds the specified dictionary by adding the upper level
    key as the "col" value to the inner dictionary.
    """
    result = []
    for param1, dicts in dict_of_dicts.items():
        for param2, values in dicts.items():
            values = values.copy()
            values[col1] = param1
            values[col2] = param2
            result.append(values)
    return result


def fold_from_list(list_of_dicts: List[Dict[str, Any]], col: str) -> dict:
    results = {}

    for d in list_of_dicts:
        d = d.copy()
        key = d.pop(col)
        results[key] = d

    return results