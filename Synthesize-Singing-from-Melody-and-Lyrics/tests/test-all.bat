@echo off
REM Copyright (c) 2023 MusicBeing Project. All Rights Reserved.
REM
REM Author: Tao Zhang <ztao8@hotmail.com>
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.

REM Get the directory containing the script file
SET "SCRIPT_DIRECTORY=%~dp0"

python -m unittest discover -s "%SCRIPT_DIRECTORY%scripts" -p "test_*.py" %*
