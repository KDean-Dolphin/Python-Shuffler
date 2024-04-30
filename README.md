# lazy-fisher-yates-shuffler

Implementation of lazy [Fisher-Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle) using a binary
tree to manage struck entries. The principal class is Shuffler, which maps an index to a value and vice versa, where the
association between an index and a value is randomly determined the first time the index-to-value mapping is requested.
The reverse value-to-index mapping is possible only after the corresponding index-to-value mapping is determined.

## Installation

```sh
pip install lazy-fisher-yates-shuffler
```

## Usage

```python
"""
Sample code.
"""

from lazy_fisher_yates_shuffler import Shuffler

# Construct a 1,000-element non-cyclic shuffler.
shuffler = Shuffler(1000, False)

# Determine the value at a given index.
index = 10
value = shuffler.value_at(index)

print(value)

# Shuffler is reversible.
assert index == shuffler.index_of(value)

# Iterate over the entire range of the shuffler (0-999).
# Values will be randomly distributed and no two will be identical.
for value in shuffler:
    print(value)

# Construct a 1,000-element cyclic shuffler.
shuffler = Shuffler(1000, True)

# Iterate over the entire range of the shuffler (0-999).
# Values will be randomly distributed and cyclic: each value is the index of the next value.
# Last value is first index (0).
for value in shuffler:
    print(value)
```
