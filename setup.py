# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
            name='komi',
            version='0.0.0',
            description='Wip : train MOGP',
            long_description='Wip : train MOGP',
            author='Team MOGP-tools',
            maintainer="Team MOGP-tools",
            author_email='olivier.truffinet@cea.fr, karim.ammar@cea.fr',
            packages=find_packages(),
            install_requires=["numpy", "pandas", "torch", "gpytorch", "pytest", "custom-profiler"],
            keywords="MOGP",
            url="https://github.com/MOGP-tools/KOMINTERN",
            python_requires=">=3.6, <4",
            #project_urls={  
            #      "Bug Reports": "https://github.com/KarGeekrie/customProfiler/issues",
            #      "Source": "https://github.com/KarGeekrie/customProfiler",
            #},
      )
