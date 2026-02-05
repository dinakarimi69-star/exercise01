from typing import Set

def id_to_fruit(fruit_id: int, fruits: Set[str]) -> str:
    """
    Returns the fruit name corresponding to the given index.

    The pervious implementation was incorrect because sets are unordered.
    I fixed the issue by converting the set to a sorted list
    before accessing the element by index.
    """
    fruits_list = sorted(fruits)

    if fruit_id < 0 or fruit_id >= len(fruits_list):
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")

    return fruits_list[fruit_id]