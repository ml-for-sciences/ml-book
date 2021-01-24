# ml-book

A short example showing how to write a lecture series using Jupyter Book 2.0.

## Creating an Conda Environment

The conda environment is provided as `environment.yml`. This environment is used for all testing by Github Actions and can be setup by:

1. `conda env create -f environment.yml`
2. `conda activate ml-book`

## Building a Jupyter Book

Run the following command in your terminal:

```bash
jb build book/
```

If you would like to work with a clean build, you can empty the build folder by running:

```bash
jb clean book/
```

If jupyter execution is cached, this command will not delete the cached folder.

To remove the build folder (including `cached` executables), you can run:

```bash
jb clean --all book/
```

## Publishing this Jupyter Book

This repository is published automatically to `gh-pages` upon `push` to the `main` branch.
