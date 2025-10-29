from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(  
    name="pep_prediction_bench",
    version="0.1.0",
    author="Li Pengyong",
    author_email="lipengyong@xidian.edu.cn",
    description="A benchmark package for peptide property prediction (classification & regression)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BioChemAI/pepbench",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.0,<2.0",
        "pandas>=1.5.0,<2.0",
        "scikit-learn>=1.2.0,<1.4.0",
        "torch>=2.0.0",
        "transformers>=4.35.0,<5.0",
        "xgboost>=1.7.0,<2.0",
        "peptidy",
    ],
    python_requires=">=3.10",
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pepbench_train=pep_prediction_bench.train:main',
            'pepbench_test=pep_prediction_bench.test:main',
        ],
    },
        classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
