"""
Persistence for lazy Fisher-Yates shuffler.

Copyright (c) 2024 by Dolphin Data Development Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
"""


from abc import ABC, abstractmethod
from typing import Final, Optional


class NodeState:
    """
    Internal node state for persistence manager.
    """

    def __init__(self, unstruck_count: int, unstruck_bitmap: int):
        """
        Initialize the node state.

        :param unstruck_count: Number of unstruck entries in or below node.

        :param unstruck_bitmap: Bitmap of unstruck entries for terminal node.
        """

        self._unstruck_count: Final[int] = unstruck_count
        """
        Number of unstruck entries in or below node.
        """

        self._unstruck_bitmap: Final[int] = unstruck_bitmap
        """
        Bitmap of unstruck entries for terminal node.
        """

    @property
    def struck_count(self) -> int:
        """
        :return: Number of unstruck entries in or below node.
        """

        return self._unstruck_count

    @property
    def struck_bitmap(self) -> int:
        """
        :return: Bitmap of unstruck entries for terminal node.
        """

        return self._unstruck_bitmap


class PersistenceManager(ABC):
    """
    Persistence manager, used to maintain shuffler state across instantiations and index/value pair mapping within and
    across instantiations.

    The use of the same persistence manager in more than one shuffler at a time or in shufflers of different constructor
    parameters across instantiations is an error.
    """

    @abstractmethod
    def save_node_state(self, key: int, node_state: NodeState):
        """
        Save node state.

        :param key: Persistence key assigned to node. Value is from 1 to 2^(N+1)-1, where "N" is the number of bits in
        size-1 and "size" is the encapsulating shuffler size.

        :param node_state: Node state.
        """

        pass

    @abstractmethod
    def restore_node_state(self, key: int) -> Optional[NodeState]:
        """
        Restore node state. If the node state was not previously saved, the node has not been previously visited and a
        None result forces the node to be constructed using default values.

        :param key: Persistence key assigned to node.

        :return: Node state if previously saved or None if not.
        """

        pass

    @abstractmethod
    def delete_node_state(self, key: int):
        """
        Delete node state. The node state for the key is known to exist.

        :param key: Persistence key assigned to node.
        """

        pass

    @abstractmethod
    def save_index_value(self, index: int, value: int):
        """
        Put an index/value pair into the underlying store.

        :param index: Index. Range is from 0 to size-1 where "size" is the encapsulating shuffler size.

        :param value: Value. For non-cyclic shufflers, range is from 0 to size-1. For cyclic shufflers, range is from
        -size to size-1.
        """

        pass

    @abstractmethod
    def delete_index_value(self, index: int, value: int):
        """
        Delete an index/value pair from the underlying store. The mapping between index and value is already known to be
        correct; both values are provided for efficient implementation.

        :param index: Index.

        :param value: Value.
        """

        pass

    @abstractmethod
    def value_at(self, index: int) -> Optional[int]:
        """
        Get the value at an index from the underlying store.

        :param index: Index.

        :return: Value or None if no mapping for the index exists.
        """

        pass

    @abstractmethod
    def index_of(self, value: int) -> Optional[int]:
        """
        Get the index for a value from the underlying store.

        :param value: Value.

        :return: Index or None if no mapping for the value exists.
        """

        pass


class MemoryPersistenceManager(PersistenceManager):
    """
    Persistence manager that maintains shuffler state and index/value pair mapping in memory.
    """

    def __init__(self):
        """
        Construct an in-memory persistence manager.
        """

        self._node_state_store: Final[dict] = dict()
        """
        Dictionary for managing node state.
        """

        self._index_value_store: Final[dict] = dict()
        """
        Dictionary for managing index/value mapping.
        """

        self._value_index_store: Final[dict] = dict()
        """
        Dictionary for managing value/index mapping.
        """

    def save_node_state(self, key: int, node_state: NodeState):
        self._node_state_store[key] = node_state

    def restore_node_state(self, key: int) -> Optional[NodeState]:
        try:
            return self._node_state_store[key]
        except KeyError:
            return None

    def delete_node_state(self, key: int):
        del self._node_state_store[key]

    def save_index_value(self, index: int, value: int):
        self._index_value_store[index] = value
        self._value_index_store[value] = index

    def delete_index_value(self, index: int, value):
        del self._value_index_store[value]
        del self._index_value_store[index]

    def value_at(self, index: int) -> Optional[int]:
        try:
            return self._index_value_store[index]
        except KeyError:
            return None

    def index_of(self, value: int) -> Optional[int]:
        try:
            return self._value_index_store[value]
        except KeyError:
            return None
