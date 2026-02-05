# exercise01
debugging exercise1
This exercise focuses on identifying and fixing a logical bug in a Python function that maps a numeric ID to a fruit name. The debugging process required understanding Python data structures and correcting incorrect assumptions about element ordering.
problem:
The method id_to_fruit() trys to return a fruit from the set based on its index (fruit_id). However, the function is returning incorrect results for the given indices. This is because sets in Python are unordered collections, which means there’s no predictable order for the elements inside a set and data structure is wrong. In Python, sets are unordered

Solution:

To fix this issue, we need to sort the set into a list before indexing it. Sorting guarantees a deterministic order and allows us to safely access fruits by their index.
