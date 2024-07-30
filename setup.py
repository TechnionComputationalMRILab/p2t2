from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()


setup(
    name="p2t2",
    version="0.1.2",
    packages=find_packages(),
    install_requires=required,
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'p2t2_train = p2t2_train:main',
            'p2t2_infer = p2t2_infer:main',
            'p2t2_simulate = p2t2_simulate:main'
        ]
    }
)
