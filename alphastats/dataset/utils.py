import functools
import http.client as httplib
import logging
import warnings
from typing import Type


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
                # TODO: make this more specific on the warning package & message!
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


def check_internetconnection():
    connection = httplib.HTTPSConnection("8.8.8.8", timeout=5)
    try:
        connection.request("HEAD", "/")
        return True
    except Exception as e:
        raise ConnectionError("No internet connection available.") from e
    finally:
        connection.close()


def check_if_df_empty(f):
    # decorator to check for missing values
    def inner(*args, **kwargs):
        if args[0].empty is True:
            raise ValueError("DataFrame is empty. No significant GO-terms found.")
        return f(*args, **kwargs)

    return inner


# TODO: unused
def list_to_tuple(function):
    """
    list are not hashable not suitable for caching
    convert to tuple
    """

    def wrapper(*args):
        args = [tuple(x) if isinstance(x, list) else x for x in args]
        result = function(*args)
        result = tuple(result) if isinstance(result, list) else result
        return result

    return wrapper


# TODO: replace with https://pandas.pydata.org/docs/reference/api/pandas.Index.has_duplicates.html#pandas.Index.has_duplicates
def find_duplicates_in_list(input_list: list) -> list:
    """Find duplicates in a list.

    Args:
        input_list (list): list

    Returns:
        list: list with duplicated values
    """
    duplicates = []
    ys = sorted(input_list)

    for n in range(1, len(input_list)):
        if ys[n] == ys[n - 1]:
            duplicates.append(ys[n])
    return duplicates
