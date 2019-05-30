#!/usr/bin/env python3
"""
Profiling tools for checking the timing of functions
and events
"""

import time
import numpy as np


class tstamp(object):
    """
    Time stamping for funciton and other events
    """

    def __init__(self):
        """
        Constructor for the start, stop, dt times
        """
    
        # Initialize the start, stop, dt, and time dict
        self.start = 0; self.stop = 0; self.dt = 0;
        self.tdict = {}

    def set_timer(self, print_str): 
        """
        Set the start timer
        """

        # Save the current time and the string associated with the time
        self.start = time.time()
        self.pstr = print_str

    def get_timer(self, print_res=True):
        """
        Get the time after a call to this function
        """

        # Get the current time, different
        self.stop = time.time()
        self.dt = self.stop - self.start 
        
        # Save the time in the dictionary
        self.tdict[self.pstr] = self.dt

        if print_res:
            print('%s: %g s' % (self.pstr, self.dt))

    def print_all_timers(self, msg=''):
        """
        Prints the contents of tdict
        """

        # Iterate over all the keys and values tdict
        print('\n%s\n' % msg)
        for k, v in self.tdict.items():
            print('%s: %g s' % (k, v))
        
        print('\nTotal: %g s\n' % sum(self.tdict.values()))
