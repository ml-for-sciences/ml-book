# ml-book

A short example showing how to write a lecture series using Jupyter Book 2.0.

## Creating an Conda Environment

The conda environment is provided as `environment.yml`. This environment is used for all testing by Github Actions and can be setup by:

1. `conda env create -f environment.yml`
2. `conda activate ml-book`

## Building a Jupyter Book

Run the following command in your terminal:

```bash
jupyter-book build /.book
```

If you would like to work with a clean build, you can empty the build folder by running:

```bash
jupyter-book clean /.book
```

If jupyter execution is cached, this command will not delete the cached folder.

To remove the build folder (including `cached` executables), you can run:

```bash
jupyter-book clean --all book/
```

## Publishing this Jupyter Book

After pushing to the main branch, go to ml-book/book and run:

```bash
ghp-import -n -p -f _build/html
```
This will push the build to the webpage branch and deploy it at [ml-lectures.org](https://ml-lectures.org/).
