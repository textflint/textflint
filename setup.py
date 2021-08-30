#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf-8') as f:
    license = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    reqs = f.read()

pkgs = [p for p in find_packages() if p.startswith('textflint')]
print(pkgs)

setup(
    name='textflint',
    version='0.0.5',
    url='https://github.com/textflint/textflint',
    description='Unified Multilingual Robustness Evaluation Toolkit '
                'for Natural Language Processing',
    long_description=readme,
    long_description_content_type='text/markdown',
    entry_points={
        "console_scripts": [
            "textflint=textflint.textflint_cli:main"],
    },
    license='GNU GENERAL PUBLIC LICENSE',
    author='Fudan NLP Team',
    author_email='xiao_wang20@fudan.edu.cn',
    python_requires='>=3.7',
    packages=pkgs,
    install_requires=reqs.strip().split('\n'),
)
