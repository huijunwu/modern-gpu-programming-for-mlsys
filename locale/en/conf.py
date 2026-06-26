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
html_logo = "../../static/mlc-logo-with-text-landscape.svg"
html_favicon = "../../static/mlc-favicon.ico"
html_static_path = ["../../static"]
html_extra_path = ["../../_extra", "../../img"]
html_css_files = ["../../static/custom.css", "../../static/demo-embed.css"]
html_js_files = ["../../static/demo-embed.js", "../../static/lang-switcher.js"]
html_theme_options = {
    # 将页面整体宽度设为 100%（默认通常为 88rem）
#     "page_width": "100%",
    # 将主要内容区域宽度设为 100%（默认通常为 75rem）
#     "content_width": "100%",
    # 如果你希望保留侧边栏固定宽度，只让内容区域自适应，可以这样组合：
    "page_width": "100%",
    "content_width": "auto",  # 让内容自动填满剩余空间
    "sidebar_width": "18rem", # 侧边栏固定宽度    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "home_page_in_toc": False,
    "use_download_button": False,
    "use_fullscreen_button": False,
}
