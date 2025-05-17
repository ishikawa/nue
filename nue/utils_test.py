# Copyright 2025 Takanori Ishikawa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for small helper functions."""

from .utils import format_number_abbrev


def test_format_number_abbrev():
    assert format_number_abbrev(1000) == "1.0K"
    assert format_number_abbrev(1000000) == "1.0M"
    assert format_number_abbrev(1000000000) == "1.0B"
