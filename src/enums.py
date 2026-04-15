from enum import StrEnum


class HtType(StrEnum):
    """HT_type attribute values written into HDF5 scan groups."""

    XRD = "xrd"
    EDX = "edx"
    MOKE = "moke"
    SEM = "sem"
    PROFIL = "profil"


class NxClass(StrEnum):
    """NX_class attribute values for NeXus-compatible HDF5 groups."""

    INSTRUMENT = "HTinstrument"
    DATA = "HTdata"
    RESULTS = "HTresults"


class H5Mode(StrEnum):
    """h5py file open modes."""

    READ = "r"
    WRITE = "w"
    APPEND = "a"
    READ_WRITE = "r+"
