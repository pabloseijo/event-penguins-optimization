"""Headless placeholder for v2e's optional input file dialog."""


def fileopenbox(*args, **kwargs):
    raise RuntimeError("The easygui file picker is unavailable in headless mode; pass --input")
