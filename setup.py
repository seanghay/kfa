import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="kfa",
    version="0.2.0",
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
        "chardet",
        "onnxruntime",
        "sosap",
        "numpy",
        "khmercut",
        "librosa",
        "requests",
        "appdirs",
    ],
    scripts=["bin/kfa"],
)
