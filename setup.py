from setuptools import setup, find_packages
import os

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import allennlp_semparse whilst setting up.
VERSION = {}
with open("nlp_formats/version.py") as version_file:
    exec(version_file.read(), VERSION)

# Load requirements.txt with a special case for allennlp so we can handle
# cross-library integration testing.
with open("requirements.txt") as requirements_file:
    import re

    def fix_url_dependencies(req: str) -> str:
        """Pip and setuptools disagree about how URL dependencies should be handled."""
        m = re.match(
            r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git", req
        )
        if m is None:
            return req
        else:
            return f"{m.group('name')} @ {req}"

    install_requirements = []
    for line in requirements_file:
        line = line.strip()
        if line.startswith("#") or len(line) <= 0:
            continue
        install_requirements.append(line)

    install_requirements = [fix_url_dependencies(req) for req in install_requirements]

setup(
    name="nlp_formats",
    version=VERSION["VERSION"],
    description=("NLP formats as abstract nlp.GeneratorBasedBuilder implementations for the huggingface/nlp framework"),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="allennlp NLP deep learning machine reading semantic parsing parsers",
    url="https://github.com/ArneBinder/nlp-formats",
    author="Arne Binder",
    author_email="arne.binder@dfki.de",
    license="Apache",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"],),
    install_requires=install_requirements,
    #include_package_data=True,
    #package_data={
    #    "allennlp_models": ["modelcards/*.json", "structured_prediction/tools/srl-eval.pl"]
    #},
    python_requires=">=3.6.1",
    zip_safe=False,
)
