#!/usr/bin/env python3
"""
Utility functions for tests

"""

import sys, os


def set_path():
    """
    Sets the path based on the current working directory
    """
    cwd = os.getcwd()
    vslq_path = '%s/../VSLQ' % cwd
    
    if vslq_path not in sys.path:
        sys.path.insert(0, vslq_path)
