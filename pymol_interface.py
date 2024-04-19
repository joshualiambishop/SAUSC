

from typing import Any, Callable, Type, TypeAlias, TypeVar
import utils


# Pymol unfortunately passes all arguments to functions as strings
PymolBool: TypeAlias = str
PymolTupleFloat: TypeAlias = str
PymolInt: TypeAlias = str
PymolFloat: TypeAlias = str


def _str_to_bool(string: PymolBool) -> bool:
    if string.lower() == "true":
        return True
    if string.lower() == "false":
        return False
    raise TypeError(f"Input {string} must be True or False")


def _str_to_tuple_float(string: PymolTupleFloat) -> tuple[float, ...]:
    # Assuming input is of the form "(1.0, 1.0, 1.0)"
    stripped = string.strip("()")
    components = stripped.split(",")
    if not all([utils.is_floatable(possible_float) for possible_float in components]):
        raise ValueError(f"Components {components} are not all floatable.")
    return tuple([float(number) for number in components])


def _str_to_float(string: str) -> float:
    if utils.is_floatable(string):
        return float(string)
    else:
        raise ValueError(f"Argument {string} could not be interpreted as a float.")


def _str_to_int(string: str) -> float:
    return int(_str_to_float(string))


pymol_convertors: dict[type, Callable] = {
    bool: _str_to_bool,
    int: _str_to_int,
    float: _str_to_float,
    tuple[float]: _str_to_tuple_float,
}

SameAsInput = TypeVar("SameAsInput")


def convert_from_pymol(argument: Any, requested_type: Type[SameAsInput]) -> SameAsInput:
    assert (
        requested_type in pymol_convertors
    ), f"Haven't implemented a conversion for type {requested_type}."
    convertor = pymol_convertors[requested_type]
    return convertor(argument)



