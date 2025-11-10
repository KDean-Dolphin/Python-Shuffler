"""
Implementation of lazy Fisher-Yates shuffle using a binary tree to manage struck entries. The principal class is
Shuffler, which maps an index to a value and vice versa, where the association between an index and a value is randomly
determined the first time the index-to-value mapping is requested. The reverse value-to-index mapping is possible only
after the corresponding index-to-value mapping is determined.

Copyright (c) 2024 Dolphin Data Development Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
"""


from __future__ import annotations

from random import Random
from typing import Final, Optional

from .bit_manager import BitManager
from .persistence import NodeState, PersistenceManager, MemoryPersistenceManager


class _Node:
    """
    Node in binary tree for lazy Fisher-Yates shuffle.
    """

    def __init__(self, shuffler: Shuffler, key: int, bit_number: int, restore: bool):
        """
        Construct a node.

        :param shuffler: Enclosing shuffler.

        :param key: Persistence key for node state.

        :param bit_number: Bit number supported by this node.

        :param restore: If true, attempts to restore the node state.
        """

        self.shuffler: Final[Shuffler] = shuffler
        """
        Enclosing shuffler.
        """

        self.key: Final[int] = key
        """
        Persistence key for node state.
        """

        self.bit_number: Final[int] = bit_number
        """
        Bit number supported by this node.
        """

        # Terminal node supports up to TERMINAL_SIZE entries.
        self.terminal: Final[bool] = bit_number == Shuffler.TERMINAL_SIZE_BIT_COUNT - 1
        """
        True if this node is a terminal node.
        """

        self.struck_count: int
        """
        Number of struck entries in or below this node.
        """

        self.struck_bitmap: int
        """
        Bitmap of struck entries if this is a terminal node.
        """

        # Attempt to restore the node state if indicated.
        node_state: Final[Optional[NodeState]] = shuffler.persistence_manager.restore_node_state(key) \
            if restore else \
            None

        if node_state is None:
            # Node has no struck count.
            self.struck_count = 0
            self.struck_bitmap = 0
        else:
            self.struck_count = node_state.struck_count
            self.struck_bitmap = node_state.struck_bitmap

        self._right: Optional[_Node] = None
        """
        Right node for managing bit = 0.
        """

        self._left: Optional[_Node] = None
        """
        Left node for managing bit = 1.
        """

    def init_right_left(self, key: int) -> _Node:
        """
        Initialize right or left node, restoring from persistence manager if self is persisted.

        :param key: Key.

        :return: Node.
        """

        # Right and/or left node will be persisted if this node has a non-zero struck count.
        return _Node(self.shuffler, key, self.bit_number - 1, self.struck_count != 0)

    @property
    def right(self) -> _Node:
        """
        :return: Right node, constructed if necessary.
        """

        if self._right is None:
            # Right key is key-2^(bit_number+1).
            self._right = self.init_right_left(self.key - self.shuffler.bit_manager.bit(self.bit_number + 1))

        return self._right

    @property
    def left(self) -> _Node:
        """
        :return: Left node, constructed if necessary.
        """

        if self._left is None:
            # Left key is key-1.
            self._left = self.init_right_left(self.key - 1)

        return self._left

    def validate_state(self, keys: list[int], size: int, cyclic: bool):
        """
        Validate node, ensuring that struck count is consistent with right and left nodes (if not terminal) or with
        struck bitmap (if terminal). Used for testing.

        :param keys:
        :param size: Expected node size.

        :param cyclic: If true, values are generated in a cyclic pattern.
        """

        keys.append(self.key)

        # Struck count must be >= 0.
        if self.struck_count < 0:
            raise Exception("Negative struck count {} at node {}".format(self.struck_count, self.key))

        node_state: Final[Optional[NodeState]] = self.shuffler.persistence_manager.restore_node_state(self.key)

        if self.struck_count == 0:
            # Nodes with zero struck count shouldn't be persisted.
            if node_state is not None:
                raise Exception("Unexpected persistence of node {} with zero struck count".format(self.key))
        else:
            # Nodes with non-zero struck count should be persisted.
            if node_state is None:
                raise Exception("Missing persistence of node {} with non-zero struck count".format(self.key))

            # Persisted node state must match in-memory node state.
            if self.struck_count != node_state.struck_count or self.struck_bitmap != node_state.struck_bitmap:
                raise Exception("Struck count {} doesn't match persisted struck count {} or struck bitmap {:08X} "
                                "doesn't match persisted struck bitmap {:08X} at node {}".
                                format(self.struck_count, node_state.struck_count,
                                       self.struck_bitmap, node_state.struck_bitmap,
                                       self.key))

        if not self.terminal:
            bit_manager: Final[BitManager] = self.shuffler.bit_manager
            bit: Final[int] = bit_manager.bit(self.bit_number)

            right_size: int
            left_size: int

            if bit_manager.is_set(size, self.bit_number + 1):
                # Size is the maximum possible and is split evenly between right and left nodes.
                right_size = bit
                left_size = bit
            elif bit_manager.is_set(size, self.bit_number):
                # Size will fill right node; put remainder in left node (clearing bit is the same as subtraction at
                # this point).
                right_size = bit
                left_size = bit_manager.clear(size, self.bit_number)
            else:
                # Size doesn't fill right node; left node will not be created.
                right_size = size
                left_size = 0

            right: Optional[_Node]
            left: Optional[_Node]

            if self.struck_count == 0:
                # Nodes with zero struck count shouldn't have right and/or left nodes in non-cyclic shuffler.
                if not cyclic and (self._right is not None or self._left is not None):
                    raise Exception("Unexpected right and/or left nodes at non-persisted node {}".format(self.key))

                # Validate nodes that have been created to support cyclic shuffler.
                right = self._right
                left = self._left
            else:
                right = self.right
                left = self.left if left_size != 0 else None

            # Left node should be None if right node can handle this node.
            if left_size == 0 and (self._left is not None or
                                   self.shuffler.persistence_manager.restore_node_state(self.key - 1) is not None):
                raise Exception("Unexpected left node at non-terminal node {}".format(self.key))

            right_left_struck_count = (right.struck_count if right is not None else 0) + \
                                      (left.struck_count if left is not None else 0)

            # Struck count should equal sum of right and left struck counts.
            if self.struck_count != right_left_struck_count:
                raise Exception("Struck count {} doesn't match sum of right and left struck counts {} at "
                                "non-terminal node {}".format(self.struck_count, right_left_struck_count,
                                                              self.key))

            # Non-terminal node should have zero bitmap.
            if self.struck_bitmap != 0:
                raise Exception("Non-zero struck bitmap at non-terminal node {}".format(self.key))

            if right is not None:
                right.validate_state(keys, right_size, cyclic)

            if left is not None:
                left.validate_state(keys, left_size, cyclic)
        else:
            # Terminal nodes shouldn't have right and/or left nodes.
            if self._right is not None or self._left is not None:
                raise Exception("Unexpected right and/or left nodes at terminal node {}".format(self.key))

            struck_bit_count: Final[int] = self.struck_bitmap.bit_count()

            # Struck count should equal number of bits in struck bitmap.
            if self.struck_count != struck_bit_count:
                raise Exception("Struck count {} doesn't match struck bit count {} at terminal node {}".format(
                    self.struck_count, struck_bit_count, self.key))


class Shuffler:
    """
    Shuffler. Values are randomly selected across a provided range using a lazy Fisher-Yates shuffle
    (https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle), where step 3 of the original method strikes the Kth
    entry not yet struck by building a binary tree on demand with each terminal node managing a 64-bit array
    representing the struck entries.

    The shuffler has two variants: non-cyclic and cyclic. The non-cyclic variant is the traditional Fisher-Yates
    shuffle, where the only requirement in the selection of the value for any unique index is that it not have been
    selected before. The cyclic variant is a lazy Sattolo algorithm, which requires that the value selected for any
    unique index be such that either it is not equal to the index for any previously generated index/value pair or that
    it link to the beginning of any open loop except its own.
    """

    TERMINAL_SIZE: Final[int] = 64
    """
    Terminal size (number of bits in qword).
    """

    _TERMINAL_SIZE_BITMASK: Final[int] = TERMINAL_SIZE - 1
    """
    Terminal size bitmask.
    """

    TERMINAL_SIZE_BIT_COUNT: Final[int] = _TERMINAL_SIZE_BITMASK.bit_count()
    """
    Terminal size bit count.
    """

    _TERMINAL_BIT_MANAGER: Final[BitManager] = BitManager(TERMINAL_SIZE)
    """
    Bit manager for terminal struck bitmap manipulation.
    """

    _BIT_COUNT_BITMASK_2: Final[int] = 0x5555555555555555
    """
    Bitmask for 2 binary digits bit count (even bits).
    """

    _BIT_COUNT_BITMASK_4: Final[int] = 0x3333333333333333
    """
    Bitmask for 4 binary digits bit count (even 2-bit groups).
    """

    _BIT_COUNT_BITMASK_8: Final[int] = 0x0F0F0F0F0F0F0F0F
    """
    Bitmask for 8 binary digits bit count (even nibbles).
    """

    def __init__(self, size: int, cyclic: bool, persistence_manager: Optional[PersistenceManager] = None):
        """
        Construct a shuffler.

        :param cyclic: If true, values are generated in a cyclic pattern.

        :param size: Size. Inputs and values range from 0 to size-1.

        :param persistence_manager: Persistence manager; defaults to memory persistence manager if None.
        """

        self._cyclic: Final[bool] = cyclic
        """
        If true, values are generated in a cyclic pattern.
        """

        self._random: Final[Random] = Random()
        """
        Randomizer.
        """

        self._persistence_manager: Final[PersistenceManager] = persistence_manager \
            if persistence_manager is not None else \
            MemoryPersistenceManager()
        """
        Persistence manager.
        """

        self._bit_manager: Optional[BitManager] = None
        """
        Bit manager up to size bit length + 1 for fast bit manipulation.
        """

        self._size = size
        """
        Size.
        """

        self._root: Optional[_Node] = None
        """
        Root.
        """

        self._remaining_size = 0
        """
        Remaining size, equivalent to size minus root struck count.
        """

        self._build_root()

    @property
    def size(self) -> int:
        """
        :return: Size.
        """

        return self._size

    #
    @property
    def cyclic(self) -> bool:
        """
        :return: If true, values are generated in a cyclic pattern.
        """

        return self._cyclic

    @property
    def persistence_manager(self) -> PersistenceManager:
        """
        :return: Persistence manager.
        """

        return self._persistence_manager

    @property
    def bit_manager(self) -> BitManager:
        """
        :return: Bit manager.
        """

        return self._bit_manager

    def _build_root(self):
        """
        Build the root based on the size.
        """

        resizing: Final[bool] = self._root is not None

        # If size is an exact power of two, subtracting 1 will force the bit length down by 1; round up to value for
        # full capacity terminal node if necessary.
        size_bit_length: Final[int] = max((self.size - 1).bit_length(), Shuffler.TERMINAL_SIZE_BIT_COUNT)

        # New root must be created if not resizing (no root yet) or root bit number is changing.
        if not resizing or size_bit_length != self._root.bit_number + 1:
            # Bit manager is aligned with number of bits in root.
            self._bit_manager = BitManager(size_bit_length + 1)

            # Root key is 2^(size_bit_length+1)-1.
            new_root: Final[_Node] = _Node(self, self.bit_manager.all_bits, size_bit_length - 1, not resizing)

            if resizing:
                # Copy struck count from old root.
                root_struck_count: Final[int] = self._root.struck_count

                # Nodes down the right from new root to above old root have to be properly constructed and persisted if
                # struck count is non-zero.
                if root_struck_count != 0:
                    update_node: Optional[_Node] = new_root

                    while update_node is not None:
                        update_node.struck_count = root_struck_count

                        self.save_node_state(update_node)

                        # Update node is never terminal so there is always a right node.
                        update_node = update_node.right

                        # If right node has non-zero struck count, it's the old root and no further adjustments are
                        # required.
                        if update_node.struck_count != 0:
                            update_node = None

            self._root = new_root

        # Cache remaining size for performance.
        self._remaining_size = self.size - self._root.struck_count

    def save_node_state(self, node: _Node):
        """
        Save node state to the persistence manager.

        :param node: Node.
        """

        self.persistence_manager.save_node_state(node.key, NodeState(node.struck_count, node.struck_bitmap))

    def _next_value(self):
        """
        Generate the next value.

        :return: Next value, randomly selected from remaining unstruck entries.
        """

        incremental_value = self._random.randrange(0, self._remaining_size)

        self._remaining_size -= 1

        value = 0

        node: Optional[_Node] = self._root

        while node is not None:
            pending_save_node = node

            # One entry in current node or a descendant will be struck.
            node.struck_count += 1

            if not node.terminal:
                right = node.right

                # Normalize output to right node by adding number of struck entries in right node.
                right_normalized_output = incremental_value + right.struck_count

                # Index is in range of right node if current node's bit is clear.
                if self.bit_manager.is_clear(right_normalized_output, node.bit_number):
                    node = right
                else:
                    # Clear bit on incremental value and set bit on value to account for right normalized value.
                    incremental_value = self.bit_manager.clear(right_normalized_output, node.bit_number)
                    value = self.bit_manager.set(value, node.bit_number)

                    node = node.left
            else:
                terminal_incremental_value_remaining = incremental_value
                terminal_value: int

                # These variables record the individual steps, except the last, of the bit count algorithm from Figure
                # 5-2 of Hacker's Delight, 2nd edition, by Henry S. Warren, Jr., to count the bits in the unstruck
                # bitmap.

                # Each variable contains the count of the named number of bits repeating up to the size of a long
                # integer (e.g., bitCount4 contains 16 counts of 4-bit sets).

                # Need to find an unstruck bit, so invert the struck bitmap to begin.
                bit_count_1 = node.struck_bitmap ^ Shuffler._TERMINAL_BIT_MANAGER.all_bits
                bit_count_2 = bit_count_1 - (bit_count_1 >> 0x01 & Shuffler._BIT_COUNT_BITMASK_2)
                bit_count_4 = ((bit_count_2 & Shuffler._BIT_COUNT_BITMASK_4) +
                              (bit_count_2 >> 0x02 & Shuffler._BIT_COUNT_BITMASK_4))
                bit_count_8 = (bit_count_4 + (bit_count_4 >> 0x04)) & Shuffler._BIT_COUNT_BITMASK_8
                bit_count_16 = bit_count_8 + (bit_count_8 >> 0x08)
                bit_count_32 = bit_count_16 + (bit_count_16 >> 0x10)

                # Starting with the broadest bit count, each block does the following:
                #
                #   1. Shifts the bit count to the right to accommodate the shift, stored in the terminal index,
                #      determined by the previous steps (does not apply to the first block).
                #   2. Extracts the right-most bit count and compares it to the incremental index remaining.
                #   3. If less than or equal, the incremental index remaining is to the left, so:
                #       a. subtracts the right-most bit count from the incremental index remaining; and
                #       b. adds the number of bits considered (using bitwise "or" as all numbers are powers of 2) to the
                #          terminal index (first block does straight assignment).
                #
                # Although a loop would yield cleaner code, the unrolled loop is faster.

                right_bit_count: int

                # 32-bit bit count; mask limits count to 0-32.
                right_bit_count = bit_count_32 & 0x3F
                if right_bit_count <= terminal_incremental_value_remaining:
                    terminal_incremental_value_remaining -= right_bit_count
                    terminal_value = 0x20
                else:
                    terminal_value = 0x00

                # 16-bit bit count; mask limits count to 0-16.
                right_bit_count = bit_count_16 >> terminal_value & 0x1F
                if right_bit_count <= terminal_incremental_value_remaining:
                    terminal_incremental_value_remaining -= right_bit_count
                    terminal_value |= 0x10

                # 8-bit bit count; mask limits count to 0-8.
                right_bit_count = bit_count_8 >> terminal_value & 0x0F
                if right_bit_count <= terminal_incremental_value_remaining:
                    terminal_incremental_value_remaining -= right_bit_count
                    terminal_value |= 0x08

                # 4-bit bit count; mask limits count to 0-4.
                right_bit_count = bit_count_4 >> terminal_value & 0x07
                if right_bit_count <= terminal_incremental_value_remaining:
                    terminal_incremental_value_remaining -= right_bit_count
                    terminal_value |= 0x04

                # 2-bit bit count; mask limits count to 0-2.
                right_bit_count = bit_count_2 >> terminal_value & 0x03
                if right_bit_count <= terminal_incremental_value_remaining:
                    terminal_incremental_value_remaining -= right_bit_count
                    terminal_value |= 0x02

                # 1-bit bit count; mask limits count to 0-1.
                right_bit_count = bit_count_1 >> terminal_value & 0x01
                if right_bit_count <= terminal_incremental_value_remaining:
                    terminal_incremental_value_remaining -= right_bit_count
                    terminal_value |= 0x01

                assert terminal_incremental_value_remaining == 0, "Terminal incremental output remaining is not zero"

                # Set bit to mark entry as struck.
                node.struck_bitmap = Shuffler._TERMINAL_BIT_MANAGER.set(node.struck_bitmap, terminal_value)

                # Add terminal value to cumulative value.
                value |= terminal_value

                node = None

            self.save_node_state(pending_save_node)

        return value

    def value_at(self, index: int) -> int:
        """
        Get the value at a given index. If the index is found in the persistence manager, the corresponding value is
        returned. Otherwise, a unique value is randomly generated and returned.

        :param index: Index.

        :return: Value at index.
        """

        if not 0 <= index < self._size:
            raise Exception("Index {} must be >=0 and < {}".format(index, self._size))

        # Check for previous generation of value.
        value: Optional[int] = self._persistence_manager.value_at(index)

        randomized: bool

        if not self._cyclic:
            randomized = value is None

            if randomized:
                value = self._next_value()
        else:
            # Value is None if index is the start of a new loop, negative if index is the end of an existing loop.
            randomized = value is None or value < 0

            if randomized:
                loop_start: int
                not_loop_start: int

                if value is None:
                    # Index is the start of a new loop.
                    loop_start = index
                    not_loop_start = ~index
                else:
                    # Index is the end of an existing loop.
                    loop_start = ~value
                    not_loop_start = value

                    # Delete existing index/value pair.
                    self.persistence_manager.delete_index_value(index, value)

                terminal_bit_number: Final[int] = loop_start & Shuffler._TERMINAL_SIZE_BITMASK

                reserved_nodes: list[_Node] = []

                node: Optional[_Node] = self._root

                while node is not None:
                    pending_save_node = node

                    reserved_nodes.append(node)

                    # One entry in current node or a descendant is being reserved.
                    node.struck_count += 1

                    if not node.terminal:
                        # Loop start is in range of right node if current node's bit is clear.
                        node = node.right \
                            if self.bit_manager.is_clear(loop_start, node.bit_number) else \
                            node.left
                    else:
                        # Set bit to mark entry as struck.
                        node.struck_bitmap = Shuffler._TERMINAL_BIT_MANAGER.set(node.struck_bitmap, terminal_bit_number)

                        node = None

                    self.save_node_state(pending_save_node)

                # Account for reserve.
                self._remaining_size -= 1

                # Store reserved remaining size (faster than incrementing after unreserve).
                reserved_remaining_size: Final[int] = self._remaining_size

                # Remaining size is zero if final loop is being closed.
                if self._remaining_size != 0:
                    value = self._next_value()

                    # Persistence manager returns None if value is the end of its loop, otherwise value is the start of
                    # an existing loop that will be joined to this one.
                    loop_end = self._persistence_manager.index_of(~value)
                    if loop_end is None:
                        loop_end = value

                    # Join loop end to loop start with not bit value to indicate incompleteness.
                    self._persistence_manager.save_index_value(loop_end, not_loop_start)

                    for reserved_node in reserved_nodes:
                        # One entry in current node or a descendant is being unreserved.
                        reserved_node.struck_count -= 1

                        if reserved_node.terminal:
                            # Clear bit to mark entry as unreserved.
                            reserved_node.struck_bitmap = Shuffler._TERMINAL_BIT_MANAGER.clear(
                                reserved_node.struck_bitmap, terminal_bit_number)

                        if reserved_node.struck_count != 0:
                            self.save_node_state(reserved_node)
                        else:
                            self.persistence_manager.delete_node_state(reserved_node.key)

                    self._remaining_size = reserved_remaining_size
                else:
                    # Close the loop.
                    value = loop_start

        if randomized:
            # Store index/value pair.
            self._persistence_manager.save_index_value(index, value)

        assert 0 <= value < self._size, "Value {} must be >=0 and < {}".format(value, self._size)

        return value

    def index_of(self, value: int) -> Optional[int]:
        """
        Get the index for a given value. If the value is found in the persistence manager, the corresponding index is
        returned. Otherwise, None is returned.

        :param value: Value.

        :return: Index of value or None if value has not yet been generated.
        """

        return self._persistence_manager.index_of(value)

    def resize(self, new_size: int):
        """
        Resize the input. Care should be taken with this operation as it will make multiple calls to the persistence
        manager to rebuild the internal state for the new size.

        :param new_size: New size.
        """

        # Ignore if no change.
        if new_size != self.size:
            if new_size < self.size and self._root.struck_count != 0:
                raise Exception("Cannot shrink a partially used shuffler")

            if self.cyclic and self._remaining_size == 0:
                raise Exception("Cannot resize a completed cyclic shuffler")

            self._size = new_size

            self._build_root()

    def validate_state(self):
        """
        Validate the entire tree, ensuring that all struck counts and bitmaps are as expected. Used for testing.
        """

        if self._remaining_size != self.size - self._root.struck_count:
            raise Exception("Remaining size {} doesn't equal input size {} minus root struck count {}".format(
                self._remaining_size, self.size, self._root.struck_count))

        keys: list[int] = []

        self._root.validate_state(keys, self.size, self.cyclic)

        keys.sort()

        minimum_key = keys[0]
        maximum_key = minimum_key - 1

        for key in keys:
            if key == maximum_key:
                raise Exception("Duplicate key {}".format(key))

            maximum_key = key

        if minimum_key < Shuffler._TERMINAL_SIZE_BITMASK << 1 | 1:
            raise Exception("Invalid minimum key {}".format(minimum_key))

        if maximum_key != minimum_key and maximum_key >= 1 << ((self.size - 1).bit_length() + 1):
            raise Exception("Invalid maximum key {}".format(maximum_key))

    class Iterator:
        """
        Iterator overlay for shuffler.
        """

        def __init__(self, shuffler: Shuffler):
            """
            Construct an iterator.

            :param shuffler: Shuffler over which to iterate.
            """

            self._shuffler: Final[Shuffler] = shuffler
            """
            Shuffler.
            """

            self._next_index = 0
            """
            Next index for which to get value.
            """

        def __next__(self):
            """
            :return: Next value in the iteration.
            """

            if self._next_index == self._shuffler.size:
                raise StopIteration

            value: Final[int] = self._shuffler.value_at(self._next_index)

            if not self._shuffler.cyclic:
                # Iterator for non-cyclic shuffler increments index.
                self._next_index += 1
            elif value != 0:
                # Iterator for cyclic shuffler moves index to the previous value in the cycle.
                self._next_index = value
            else:
                # End of cyclic shuffle.
                self._next_index = self._shuffler.size

            return value

    def __iter__(self):
        """
        Get an iterator for self. For non-cyclic values, the results are the equivalent of calling value_at() on each
        index in order from 0 to size-1. For cyclic values, the results are the equivalent of calling value_at() on the
        result of the previous iteration (initially 0) until the cycle is complete.

        :return: Shuffler iterator.
        """

        return Shuffler.Iterator(self)
