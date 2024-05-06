from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="p2t2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'p2t2_train = p2t2_train:main',
            'p2t2_infer = p2t2_infer:main',
            'p2t2_simulate = p2t2_simulate:main'
        ]
    }
)
