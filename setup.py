from setuptools import setup, find_packages

DEPENDENCIES = [
    "numpy>=1.16",
    "scipy>=1.0",
    "pandas>=1.0",
    "h5py>=3.0",
    "matplotlib>=3.0",
    "pymusic>=0.1.0",
]

setup(
    name="music_scripts",
    version="0.1.0",
    description="MUSIC toolshed",
    url="https://github.com/amorison/music_scripts/",
    author="Adrien Morison",
    author_email="adrien.morison@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=DEPENDENCIES,
    package_data={"music_scripts": ["py.typed"]},
)
