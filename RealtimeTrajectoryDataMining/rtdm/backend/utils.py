import random
import time
from functools import wraps
from typing import List, Tuple, Union

import pandas as pd  # pip install pandas
from backend.typealias import CoordinatePair

# General helper methods


def get_queue(namespace: str, user: str, name: str) -> str:
    """
    Construct a proper key for accessing a certain datastructure in Redis.

    Args:
        namespace (str): A namespace used to e.g. identify the class that
            manages the list/queue.
        user (str): A unique identifier of a user.
        name (str): The name of the list/queue that represent a certain step
            in the data-processing pipleline.

    Returns:
        str: The key to a list/queue in Redis.
    """
    return f"{namespace}:{user}:{name}"


def rand_24_bit() -> int:
    """Returns a random 24-bit integer."""
    return random.randrange(0, 16 ** 6)  # noqa


def color_dec() -> int:
    """Alias of rand_24 bit()"""
    return rand_24_bit()


def color_hex(num: Union[None, int] = None) -> str:
    """Returns a 24-bit int in hex."""
    if num is None:
        num = rand_24_bit()
    return "%06x" % num


def color_rgb(num: Union[None, int] = None) -> Tuple[int, int, int]:
    """Returns three 8-bit numbers, one for each channel in RGB."""
    if num is None:
        num = rand_24_bit()
    hx = color_hex(num)
    barr = bytearray.fromhex(hx)
    return (barr[0], barr[1], barr[2])


def str_is_number(string: str) -> bool:
    """
    Check whether a given string input is a string or actually a number.

    Args:
        s (str): A string.

    Returns:
        (bool): True if the string can be cast as a float (it is actually a \
            number). False if the string can not be cast as a float (it is \
            actually not a number).
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def timing(function):
    """
    Define a decorator function to time a function call.

    Note:
        The function f is called in an ordinary manner and the result is \
        returned. The time of the function call is simply logged to stdout \
        via the print() function.

    Args:
        f (function): The function to be timed.

    Returns:
        wrapper (function): The result of the function f.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        start_sysusr = time.process_time()
        start_wall = time.time()
        result = function(*args, **kwargs)
        end_sysusr = time.process_time()
        end_wall = time.time()
        print(
            "INFO: Call to "
            + str(function)
            + ". Elapsed time (sysusr): "
            + str(end_sysusr - start_sysusr)
            + " Elapsed time (wall): "
            + str(end_wall - start_wall)
        )
        return result

    return wrapper


# Pandas dataframe-specific helper methods


def filter_dataframe(
    df: pd.DataFrame, bounds: List[CoordinatePair]
) -> pd.DataFrame:
    """
    Filter a dataframe based on latitude/longitude upper and lower bounds.

    Args:
        df (pd.DataFrame): A dataframe containing timestamped location data.
        bounds (List[CoordinatePair]): Upper and lower latitude/longitude \
            bounds.

    Raises:
        KeyError: If a required column is not present in the given dataframe.
        ValueError: If a latitude upper bound is smaller than a lower bound.
        ValueError: If a longitude upper bound is smaller than a lower bound.

    Returns:
        pd.DataFrame: A filtered dataframe containing latitude/longitude data \
            that falls within given upper and lower bounds.
    """
    for column_name in ["longitude", "latitude", "external_timestamp"]:
        if column_name not in df.columns:
            raise KeyError(
                f"Column name {column_name} is not present in the dataframe!"
            )
    if not bounds[1][0] < bounds[0][0]:
        raise ValueError(
            f"Latitude upper bound 'bounds[1][0]' = {bounds[1][0]} is smaller \
            than the lower bound 'bounds[0][0]' = {bounds[0][0]}."
        )
    if not bounds[1][0] < bounds[0][0]:
        raise ValueError(
            f"Longitude upper bound 'bounds[1][1]' = {bounds[1][1]} is smaller \
            than the lower bound 'bounds[0][1]' = {bounds[0][1]}."
        )
    df_filtered = df[
        (df["latitude"] < bounds[0][0])
        & (df["latitude"] > bounds[1][0])
        & (df["longitude"] < bounds[0][1])
        & (df["longitude"] > bounds[1][1])
    ]
    return df_filtered.sort_values(by=["external_timestamp"])
