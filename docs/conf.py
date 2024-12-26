# -*- coding: utf-8 -*-
#
# Triton documentation build configuration file, created by
# sphinx-quickstart on Mon Feb 10 01:19:09 2020.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

import os
import platform
import shutil
import sys
import sysconfig
from pathlib import Path

import sphinx_rtd_theme
from sphinx_gallery.sorting import FileNameSortKey


def process_sig(app, what, name, obj, options, signature, return_annotation):
    if signature and "_builder" in signature:
        signature = signature.split("_builder")[0] + ")"
    return (signature, return_annotation)


def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path("../python") / "build" / dir_name
    return cmake_dir


def setup_generated_mlir_docs():
    dst_path = Path("dialects")
    os.makedirs(dst_path, exist_ok=True)

    cmake_dir = get_cmake_dir()
    src_dir = cmake_dir / "docs" / "dialects"
    assert os.path.isdir(src_dir)

    shutil.copytree(src_dir, dst_path, dirs_exist_ok=True)

    files = os.listdir(dst_path)

    dialects = "\n   ".join(["./" + f for f in files if "Dialect" in f])
    ops = [f for f in files if "Ops" in f]

    # Add titles
    for op in ops:
        with open(dst_path / op, "r+") as f:
            lines = f.readlines()
            lines.insert(0, "# " + op.split(".md")[0])
            f.seek(0)
            f.writelines(lines)
    ops = "\n   ".join(["./" + op for op in ops])

    rst_string = f"""
Triton MLIR Dialects and Ops
=====================

.. toctree::
   :maxdepth: 1
   :caption: Dialects

   {dialects}

.. toctree::
   :maxdepth: 1
   :caption: Dialect Ops

   {ops}
"""
    with open(dst_path / "dialects.rst", "w+") as f:
        f.write(rst_string)


def setup(app):
    """Customize function args retrieving to get args under decorator."""
    import subprocess

    import sphinx

    app.connect("autodoc-process-signature", process_sig)
    max_jobs = os.getenv("MAX_JOBS", str(2 * os.cpu_count()))
    print(f"Installing Triton Python package using {max_jobs} threads")
    subprocess.run("pip install -e ../python", shell=True, env=os.environ.copy())

    setup_generated_mlir_docs()

    def forward_jit_fn(func):
        old = func

        def wrapped(obj, **kwargs):
            import triton

            if isinstance(obj, triton.runtime.JITFunction):
                obj = obj.fn
            return old(obj)

        return wrapped

    old_documenter = sphinx.ext.autosummary.get_documenter

    def documenter(app, obj, parent):
        import triton

        if isinstance(obj, triton.runtime.JITFunction):
            obj = obj.fn
        return old_documenter(app, obj, parent)

    sphinx.ext.autosummary.get_documenter = documenter
    sphinx.util.inspect.unwrap_all = forward_jit_fn(sphinx.util.inspect.unwrap_all)
    sphinx.util.inspect.signature = forward_jit_fn(sphinx.util.inspect.signature)
    sphinx.util.inspect.object_description = forward_jit_fn(sphinx.util.inspect.object_description)


# Auto Doc

sys.path.insert(0, os.path.abspath("../python/"))
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_multiversion",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
]
autosummary_generate = True

# versioning config
smv_tag_whitelist = r"^(v3.2.0)$"
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = None
smv_released_pattern = r"^tags/.*$"
smv_outputdir_format = "{ref.name}"
smv_prefer_remote_refs = False

# Sphinx gallery
extensions += ["sphinx_gallery.gen_gallery"]

sphinx_gallery_conf = {
    "examples_dirs": "../python/tutorials/",
    "gallery_dirs": "getting-started/tutorials",
    "filename_pattern": "",
    "ignore_pattern": r"(__init__\.py|11.*.py)",
    "within_subsection_order": FileNameSortKey,
    "reference_url": {
        "sphinx_gallery": None,
    },
    # Examples don't work on non-Linux platforms, because they actually run
    # Triton.  But it's nice to be able to run the rest of the docs build.
    "abort_on_example_error": platform.system() == "Linux",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_sidebars = {
    "**": [
        "_templates/versions.html",
    ],
}

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Triton"
copyright = "2020, Philippe Tillet"
author = "Philippe Tillet"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = ""
# The full version, including alpha/beta/rc tags.
release = ""

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    "**": [
        "relations.html",  # needs 'show_related': True theme option to display
        "searchbox.html",
    ]
}

html_logo = "https://cdn.openai.com/triton/assets/triton-logo.png"

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "Tritondoc"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "Triton.tex", "Triton Documentation", "Philippe Tillet", "manual"),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "triton", "Triton Documentation", [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "Triton",
        "Triton Documentation",
        author,
        "Triton",
        "One line description of project.",
        "Miscellaneous",
    ),
]
