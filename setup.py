import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="plangid",
    version="0.0.1",
    author="Santiago M. Mola",
    author_email="santi@mola.io",
    description="A programming language detector.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smola/plangid",
    packages=setuptools.find_packages(exclude=("tests",)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["click", "hyperopt", "pandas", "pyyaml", "scikit-learn",],
    tests_require=["pytest"],
)
