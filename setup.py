import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name="AIBias-Oddgeir",
        version="0.0.1",
        author="Oddgeir Georgsson",
        author_email="oddgeir.pall@gmail.com",
        description="Package for analysing and mitigating bias in datasets",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="todo",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.6",
)
