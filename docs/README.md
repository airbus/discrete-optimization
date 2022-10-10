# How to generate the documentation

## Install sphinx and needed extensions

```shell
cd docs/
pip install -r requirements.txt
```

## Generate API doc source files

This command must be launched each time a new subpackage or module is added to the code.

```shell
cd docs/
sphinx-apidoc -o source/api -f -T ../discrete_optimization
```

If a module/subpackage is removed, you have to remove the corresponding doc source file.

You can also remove all the generated files with
```shell
rm source/api/discrete_optimization*.rst
```
and then regenerate all files with the above command to be sure not to left a file
corresponding to an old version of the code.

## Generate notebooks list

We generate a file notebooks.md from a template file notebooks.template.md, by including the
list of available notebooks, with the command

```shell
cd docs/
python generate_nb_index.py
```

## Build doc

```shell
cd docs/
make html
```

The resulting doc is generated in `build/` directory.

## Guidelines when writing the documentation

### In-code docstrings

The API is self-documented thanks to in-code docstrings. In order to generate properly the API doc,
some guidelines should be followed:

- Docstrings should follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings
(see this [example](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google)),
parsed thanks to [napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html?highlight=napoleon#module-sphinx.ext.napoleon).

- As we use type annotations in the code, types hints should not be added to the docstrings in order to avoid duplicates,
and potentially inconsistencies.

- Docstrings should also follow [PEP-257](https://peps.python.org/pep-0257/) which can be checked
thanks to [pydocstyle](https://www.pydocstyle.org/en/stable/) tool.

    ```shell
    pip install pydocstyle
    pydocstyle discrete_optimization
    ```

### Doc pages

Natively, sphinx is meant to parse [reStructuredText files](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

To be able to directly reuse materials already written in markdown format, we use here the [myST extension](https://myst-parser.readthedocs.io/en/latest/intro.html),
that allows to write [makdown files](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html).

In markdown files, we can still write sphinx directives and roles as explained in [myST documentation](https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html).
For instance a sphinx table of contents tree can be inserted in a markodown file with a code block like:

    ```{toctree}
    ---
    maxdepth: 2
    caption: Contents
    ---
    install
    getting_started
    notebooks
    api/modules
    contribute
    ```
