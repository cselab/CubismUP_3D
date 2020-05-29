import os, sys
sys.path.insert(
        1, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'python')))

import unittest


class TestCaseExMetaclass(type(unittest.TestCase)):
    """Wrap built-in TestCase class with additional features.

    By default, all methods with a prefix `test_` are treated as tests.
    We make another variant:
        `etest_` prefix has a priority. If any `etest_*` methods are present,
        only them will be run, and others will be disabled.
    """
    def __new__(cls, name, parents, dct):
        if any(key.startswith('etest_') for key in dct):
            # If there is any `etest_*` member functions, rename those to
            # `test_*` and ignore any other `test_*` functions.
            new_dct = OrderedDict()
            for key, value in dct.items():
                if key.startswith('etest_'):
                    new_dct[key[1:]] = value
                    print('Restricting `{}` to `{}`.'.format(name, key))
                elif not key.startswith('test_'):
                    new_dct[key] = value
            dct = new_dct
        return super().__new__(cls, name, parents, dct)



class TestCaseEx(unittest.TestCase, metaclass=TestCaseExMetaclass):
    """Extension of builtin `unittest.TestCase` class."""
    pass
