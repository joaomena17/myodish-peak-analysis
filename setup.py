from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="myodish-peak-analysis",
    version="1.0.3",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
)