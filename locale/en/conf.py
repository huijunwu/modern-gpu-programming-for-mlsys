# Sphinx configuration for the Modern GPU Programming For MLSys book (English)
project = "Modern GPU Programming For MLSys"
author = "MLC Community"
copyright = "2026, MLC Community"
release = "0.0.1"
language = "en"

extensions = ["myst_parser", "sphinx_copybutton"]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "**/README.md",
    "_*.md",
    "**/_*.md",
    "setup.py",
    "tirx_tutorial",
    "references.bib",
    "img/scripts",
    ".git",
    ".github",
]

html_theme = "sphinx_book_theme"
html_title = project
html_logo = "mlc-logo-with-text-landscape.svg"
html_favicon = "mlc-favicon.ico"
html_static_path = ["../../static"]
html_extra_path = ["../../_extra", "../../img"]
html_css_files = ["custom.css", "demo-embed.css"]
html_js_files = ["demo-embed.js", "lang-switcher.js"]
html_theme_options = {
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "home_page_in_toc": False,
    "use_download_button": False,
    "use_fullscreen_button": False,
}
