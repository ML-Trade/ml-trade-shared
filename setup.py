import setuptools, os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ml-trade-shared',
    version='0.0.1',
    author='Kyle Doidge',
    author_email='kyle.blue.nuttall@gmail.com',
    description='Shared python libraries for ml-trade project. Includes DataPreprocessing and Models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ML-Trade/ml-trade-shared',
    project_urls = {
        "Bug Tracker": "https://github.com/ML-Trade/ml-trade-shared/issues"
    },
    license='proprietary and confidential',
    packages=['ml-trade-shared'],
    install_requires=[
        "tensorflow>=2.7.0",
        "tensorboard>=2.7.0",
        "keras>=2.7.0",
        "pandas",
        "numpy",
        "jinja2"
    ],
)