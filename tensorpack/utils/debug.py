# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
# File: debug.py


import sys


def enable_call_trace():
    """ Enable trace for calls to any function. """
    def tracer(frame, event, arg):
        if event == 'call':
            co = frame.f_code
            func_name = co.co_name
            if func_name == 'write' or func_name == 'print':
                # ignore write() calls from print statements
                return
            func_line_no = frame.f_lineno
            func_filename = co.co_filename
            caller = frame.f_back
            if caller:
                caller_line_no = caller.f_lineno
                caller_filename = caller.f_code.co_filename
                print('Call to `%s` on line %s:%s from %s:%s' %
                      (func_name, func_filename, func_line_no,
                       caller_filename, caller_line_no))
            return
    sys.settrace(tracer)


if __name__ == '__main__':
    enable_call_trace()

    def b(a):
        print(2)

    def a():
        print(1)
        b(1)

    a()
