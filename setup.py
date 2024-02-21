import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="kfa",
    version="0.1.0",
    description="Khmer Forced Aligner powered by Wav2Vec2CTC and Phonetisaurus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seanghay/kfa",
    author="Seanghay Yath",
    author_email="seanghay.dev@gmail.com",
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Natural Language :: English",
    ],
    python_requires=">3.5",
    packages=setuptools.find_packages(exclude=["bin"]),
    package_dir={"kfa": "kfa"},
    package_data={"kfa": ["g2p.fst", "lexicon.pkl"]},
    include_package_data=True,
    install_requires=[
        "chardet>=5.2.0",
        "onnxruntime>=1.16.0",
        "sosap==0.0.1",
        "numpy>=1.24.4",
        "khmercut==0.0.2",
        "librosa>=0.10.1",
        "requests>=2.31.0",
        "appdirs>=1.4.4",
    ],
    scripts=["bin/kfa"],
)
