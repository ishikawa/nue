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

"""Global tokenizer instance used across training workers."""

from sentencepiece import SentencePieceProcessor

from nue.common import BUILD_DIR

# Put global tokenizer to make child process inherit
TOKENIZER = SentencePieceProcessor()
TOKENIZER.Load(str(BUILD_DIR / "tokenizer.model"))

PAD_TOKEN_ID: int = TOKENIZER.pad_id()
IGNORE_TOKEN_ID = -100

assert PAD_TOKEN_ID is not None
assert isinstance(PAD_TOKEN_ID, int)
assert PAD_TOKEN_ID >= 0, "PAD_ID must be non-negative"
