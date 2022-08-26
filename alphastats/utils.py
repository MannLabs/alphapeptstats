import functools
import warnings
from typing import Type
import logging


def ignore_warning(warning: Type[Warning]):
    """
    Ignore a given warning occurring during method execution.

    Args:
        warning (Warning): warning type to ignore.

    Returns:
        the inner function

    """

    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=warning)
                return func(*args, **kwargs)

        return wrapper

    return inner


class LoaderError(Exception):
    """Loader Exception that will be logged."""

    def __init__(self, message):
        self.message = message
        logging.error(message)
        super().__init__(self.message)


def check_for_missing_values(f):
    # decorator to check for missing values
    def inner(*args, **kwargs):
        if args[0].mat.isna().values.any() is True:
            raise ValueError(
                "Data contains missing values. Consider Imputation:"
                "for instance `DataSet.preprocess(imputation='mean')`."
            )
        return f(*args, **kwargs)

    return inner
