"""
Shuffler unit test.

Copyright (c) 2024 Dolphin Data Development Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
"""


from typing import Final, Iterable
from unittest import TestCase

from lazy_fisher_yates_shuffler import MemoryPersistenceManager
from lazy_fisher_yates_shuffler import Shuffler


class TestShuffler(TestCase):
    size: Final[int] = 111111

    validate_interval: Final[int] = 10889

    def _generate(self, shuffler: Shuffler, iterable: Iterable, indexes: [int], values: [int]):
        validate_countdown: int = 0

        for index in iterable:
            # Validate state periodically.
            if validate_countdown == 0:
                shuffler.validate_state()
                validate_countdown = self.validate_interval

            value: int = shuffler.value_at(index)

            # Check that no duplicates have been generated.
            self.assertEqual(-1, indexes[value])
            self.assertEqual(-1, values[index])

            indexes[value] = index
            values[index] = value

            validate_countdown -= 1

        # Iteration may not cover entire size; validate at the end with whatever remaining size.
        shuffler.validate_state()

    def _compare(self, shuffler: Shuffler, iterable: Iterable, indexes: [int], values: [int]):
        for index in iterable:
            value: int = shuffler.value_at(index)

            # Check that indexes and values match.
            self.assertEqual(index, indexes[value])
            self.assertEqual(value, values[index])

    def _assert_cyclic(self, shuffler: Shuffler):
        visited: [bool] = [False] * shuffler.size

        index: int = 0

        while True:
            self.assertFalse(visited[index])

            visited[index] = True

            index = shuffler.value_at(index)

            # Break when index returns to its starting point.
            if index == 0:
                break

        for index in range(shuffler.size):
            self.assertTrue(visited[index], index)

    def _test_repeatable(self, cyclic: bool):
        shuffler: Shuffler = Shuffler(self.size, cyclic)

        indexes: [int] = [-1] * self.size
        values: [int] = [-1] * self.size

        self._generate(shuffler, range(self.size), indexes, values)
        self._compare(shuffler, range(self.size), indexes, values)

        if cyclic:
            self._assert_cyclic(shuffler)

    def test_repeatable(self):
        self._test_repeatable(False)
        self._test_repeatable(True)

    def _test_cyclic(self, sequential: bool):
        shuffler: Shuffler = Shuffler(self.size, True)

        indexes: [int] = [-1] * self.size
        values: [int] = [-1] * self.size

        self._generate(shuffler, range(self.size) if sequential else Shuffler(self.size, False), indexes, values)

        visited: [bool] = [False] * self.size
        value: int = 0

        for index in range(self.size):
            value = shuffler.value_at(value)

            self.assertFalse(visited[value])

            visited[value] = True

        self.assertEqual(0, value)

        for index in range(self.size):
            self.assertTrue(visited[index])

    def test_cyclic(self):
        self._test_cyclic(True)
        self._test_cyclic(False)

    def _test_iterable(self, cyclic: bool):
        shuffler: Shuffler = Shuffler(self.size, cyclic)

        indexes: [int] = [-1] * self.size
        values: [int] = [-1] * self.size
        index: int = 0

        shuffler.validate_state()

        for value in shuffler:
            # Check that no duplicates have been generated.
            self.assertEqual(-1, indexes[value])
            self.assertEqual(-1, values[index])

            # Check that value was generated from expected index.
            self.assertEqual(value, shuffler.value_at(index))

            indexes[value] = index
            values[index] = value

            index = index + 1 if not cyclic else value

        self.assertEqual(self.size if not cyclic else 0, index)

        self.assertNotIn(-1, indexes)
        self.assertNotIn(-1, values)

        if cyclic:
            self._assert_cyclic(shuffler)

    def test_iterable(self):
        self._test_iterable(False)
        self._test_iterable(True)

    def test_persistence_manager(self):
        size_3_4_int: int = self.size * 3 // 4

        indexes: [int] = [-1] * self.size
        values: [int] = [-1] * self.size

        persistence_manager: MemoryPersistenceManager = MemoryPersistenceManager()

        shuffler1: Shuffler = Shuffler(self.size, False, persistence_manager)

        self._generate(shuffler1, range(size_3_4_int), indexes, values)

        shuffler2: Shuffler = Shuffler(self.size, False, persistence_manager)

        self._compare(shuffler2, range(size_3_4_int), indexes, values)
        self._generate(shuffler2, range(size_3_4_int, self.size), indexes, values)
        self._compare(shuffler2, range(self.size), indexes, values)

    def _test_resize(self, cyclic: bool):
        terminal_to_non_terminal_shuffler: Shuffler = Shuffler(20, cyclic)
        terminal_to_non_terminal_shuffler.validate_state()
        terminal_to_non_terminal_shuffler.resize(200)
        terminal_to_non_terminal_shuffler.validate_state()

        test_size: int = 20

        shuffler: Shuffler = Shuffler(test_size, cyclic)
        index_value_dict: dict = dict()

        for resize_index in range(20):
            for index in range(test_size // 2, test_size):
                index_value_dict[index] = shuffler.value_at(index)

            shuffler.validate_state()

            # Add 50% to test size each time.
            test_size = test_size * 3 // 2

            shuffler.resize(test_size)

            if resize_index % 2 == 0:
                # Recreate shuffler to test persistence of resize.
                shuffler = Shuffler(test_size, cyclic, shuffler.persistence_manager)

            shuffler.validate_state()

            for index, value in index_value_dict.items():
                self.assertEqual(value, shuffler.value_at(index))
                self.assertEqual(index, shuffler.index_of(value))

        if not cyclic:
            # Consume entire range.
            for index in range(test_size):
                index_value_dict[index] = shuffler.value_at(index)

            shuffler.resize(test_size + 1)
        else:
            # Cyclic assertion will consume entire range.
            self._assert_cyclic(shuffler)

            self.assertRaises(Exception, lambda: shuffler.resize(test_size + 1))

    def test_resize(self):
        self._test_resize(False)
        self._test_resize(True)
