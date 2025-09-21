from setuptools import setup

setup(
    name='Mint_NM',
    version='0.1.0',
    description='A Jupyter widgets for solving',
    url='https://github.com/mwelland/ENGPHYS_3NM4/Mint_NM',
    author='Joel',
    packages=['Mint_NM'],
    install_requires=[
        'numpy',
        'matplotlib',
        'ipywidgets',
        'IPython',
        'pyppeteer',
        'nbconvert',
        'scipy',
        'plotly'
    ],
    python_requires='>=3.7',
    license="CC-BY-4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
)
