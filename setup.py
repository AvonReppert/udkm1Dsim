from setuptools import setup, find_packages

setup(
    name='udkm1Dsim',
    version='1.5.6',
    packages=find_packages(),
    package_data={
        'udkm1Dsim': ['parameters/atomic_form_factors/chantler/*.cf',
                      'parameters/atomic_form_factors/chantler/*.md',
                      'parameters/atomic_form_factors/henke/*.nff',
                      'parameters/atomic_form_factors/henke/*.md',
                      'parameters/atomic_form_factors/cromermann.txt',
                      'parameters/magnetic_form_factors/*.mf',
                      'parameters/elements.dat',
                      'matlab/*.m',
                      ],
    },
    url='https://github.com/dschick/udkm1Dsim',
    install_requires=['tqdm>=4.43.0',
                      'numpy>=1.18.2',
                      'pint>=0.9',
                      'scipy>=1.4.1',
                      'sympy>=1.5.1',
                      'tabulate',
                      'matplotlib>=2.0.0'],
    extras_require={
        'parallel':  ['dask[distributed]>=2.6.0'],
        'testing': ['flake8', 'pytest'],
        'documentation': ['sphinx', 'nbsphinx', 'sphinxcontrib-napoleon',
                          'autodocsumm'],
    },
    license='MIT',
    author='Daniel Schick',
    author_email='schick.daniel@gmail.com',
    description='A Python Simulation Toolkit for 1D Ultrafast Dynamics '
                + 'in Condensed Matter',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.5',
    keywords='ultrafast dynamics condensed matter 1D',
)
