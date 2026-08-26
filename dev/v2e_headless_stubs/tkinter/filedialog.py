"""Headless placeholder for v2e's unused tkinter file dialog."""


def askopenfilename(*args, **kwargs):
    raise RuntimeError("The tkinter file picker is unavailable in headless mode; pass --input")
