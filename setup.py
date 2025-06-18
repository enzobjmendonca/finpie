from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="finpie",
    version="0.1.31",
    author="Enzo Mendonça",
    author_email="enzobjmendonca@gmail.com",
    description="A comprehensive Python library for Brazilian financial data analysis and quantitative research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enzobjmendonca/finpie",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.25.0",
            "myst-parser>=2.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
            "ipykernel>=6.0.0",
        ],
    },
) 