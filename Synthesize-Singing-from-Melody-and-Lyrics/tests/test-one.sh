#!/bin/sh

# Copyright (c) 2023 MusicBeing Project. All Rights Reserved.
#
# Author: Tao Zhang <ztao8@hotmail.com>
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

# Get the directory containing the script file
SCRIPT_DIRECTORY=$(dirname "$0")

if [ -z "$1" ]; then
    echo "Usage: $0 <test_python_file> [unittest_args]"
    exit 1
fi

python -m unittest discover -s $SCRIPT_DIRECTORY/scripts -p "$1" "${@:2}"