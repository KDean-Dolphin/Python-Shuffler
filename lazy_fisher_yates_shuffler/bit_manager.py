"""
Bit manager for lazy Fisher-Yates shuffler.

Copyright (c) 2024 by Dolphin Data Development Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
"""


from typing import Final


class BitManager:
    """
    Bit manager that provides bit-centric values and functions up to a specified bit count.
    """

    def __init__(self, bit_count: int):
        """
        Construct a bit manager.

        :param bit_count: Bit count. This is the total number of bits supported by this bit manager, with all values
        ranging from 0 to 2^bit_count - 1.
        """

        self._bit_count: Final[int] = bit_count
        """
        Bit count.
        """

        self._bitmasks = [0] * bit_count
        """
        Bitmasks. The value at any index has all bits up to and including the bit number represented by that index set.
        """

        self._not_bitmasks = [0] * bit_count
        """
        Not bitmasks. The value at any index has all bits except those up to and including the bit number represented by
        that index set.
        """

        self._bits = [0] * bit_count
        """
        Bits. The value at any index has the bit at the bit number represented by that index set.
        """

        self._not_bits = [0] * bit_count
        """
        Not bits. The value at any index has all bits set except the bit number represented by that index.
        """

        bitmask = 0
        bit = 1

        # Initialize bitmasks and bits.
        for bit_number in range(bit_count):
            bitmask |= bit

            self._bitmasks[bit_number] = bitmask
            self._bits[bit_number] = bit

            bit <<= 1

        self._all_bits: Final[int] = bitmask
        """
        Value of all bits set. This is equal to the last entry in the bitmasks array and is replicated for convenience.
        """

        # Initialize not bitmasks and not bits.
        for bit_number in range(bit_count):
            self._not_bitmasks[bit_number] = self._bitmasks[bit_number] ^ bitmask
            self._not_bits[bit_number] = self._bits[bit_number] ^ bitmask

    @property
    def bit_count(self) -> int:
        """
        :return: Bit count.
        """

        return self._bit_count

    @property
    def all_bits(self) -> int:
        """
        :return: Value of all bits set.
        """

        return self._all_bits

    def bitmask(self, bit_number: int) -> int:
        """
        :param bit_number: Bit number.

        :return: Bitmask with all bits up to and including bit number set.
        """

        return self._bitmasks[bit_number]

    def not_bitmask(self, bit_number: int) -> int:
        """
        :param bit_number: Bit number.

        :return: Not bitmask with all bits except those up to and including bit number set.
        """

        return self._not_bitmasks[bit_number]

    def mask_to(self, value: int, bit_number: int) -> int:
        """
        Apply bitmask to value.

        :param value: Value.
        :param bit_number: Bit number.

        :return: Value with all bits above bit number cleared.
        """

        return value & self.bitmask(bit_number)

    def mask_from(self, value: int, bit_number: int) -> int:
        """
        Apply not bitmask to value.

        :param value: Value.
        :param bit_number: Bit number.

        :return: Value with all bits at and below bit number cleared.
        """

        return value & self.not_bitmask(bit_number)

    def bit(self, bit_number: int) -> int:
        """
        :param bit_number: Bit number.

        :return: Bit at bit number.
        """

        return self._bits[bit_number]

    def not_bit(self, bit_number: int) -> int:
        """
        :param bit_number: Bit number.

        :return: All bits except that at bit number set.
        """

        return self._not_bits[bit_number]

    def is_set(self, value: int, bit_number: int) -> bool:
        """
        Determine if a specified bit is set.

        :param value: Value.
        :param bit_number: Bit number.

        :return: True if bit at bit number in value is set.
        """

        return value & self.bit(bit_number) != 0

    def is_clear(self, value: int, bit_number: int) -> bool:
        """
        Determine if a specified bit is clear.

        :param value: Value.
        :param bit_number: Bit number.

        :return: True if bit at bit number in value is clear.
        """

        return value & self.bit(bit_number) == 0

    def set(self, value: int, bit_number: int) -> int:
        """
        Set a specified bit.

        :param value: Value.
        :param bit_number: Bit number.

        :return: Value with bit at bit number set.
        """

        return value | self.bit(bit_number)

    def clear(self, value: int, bit_number: int) -> int:
        """
        Clear a specified bit.

        :param value: Value.
        :param bit_number: Bit number.

        :return: Value with bit at bit number clear.
        """

        return value & self.not_bit(bit_number)

    def toggle(self, value: int, bit_number: int) -> int:
        """
        Toggle a specified bit.

        :param value: Value.
        :param bit_number: Bit number.

        :return: Value with bit at bit number toggled.
        """

        return value ^ self.bit(bit_number)
