# Copyright (c) 2024 MusicBeing Project. All Rights Reserved.
#
# Author: Feee <cgoxopx@outlook.com>

from setuptools import setup, find_packages  
  
with open('requirements.txt') as f:  
    requirements = f.read().splitlines()  
  
setup(  
    name='your_project_name',  
    version='0.1',  
    packages=find_packages(),  
    install_requires=requirements,   
)