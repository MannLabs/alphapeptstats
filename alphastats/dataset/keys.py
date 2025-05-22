import traceback

"""String constants for accessing columns."""


class ConstantsClass(type):
    """A metaclass for classes that should only contain string constants."""

    def __setattr__(self, name, value):
        stack = traceback.format_stack()
        if any("cloudpickle" in line for line in stack):
            # Allow setting attributes when pickling
            super().__setattr__(name, value)
            return
        raise TypeError("Constants class cannot be modified")

    def get_values(cls):
        """Get all user-defined string values of the class."""
        return [
            value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and isinstance(value, str)
        ]


class Cols(metaclass=ConstantsClass):
    """String constants for accessing columns of the main dataframe in DataSet."""

    INDEX = "index_"
    GENE_NAMES = "gene_names_"
    SAMPLE = "sample_"
