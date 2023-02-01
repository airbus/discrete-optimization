#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

# Thanks to :
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
# Timeout decorator working on all platform contrary to signal library

from __future__ import print_function

import logging
import sys
import threading
from time import sleep

try:
    import thread
except ImportError:
    import _thread as thread

logger = logging.getLogger(__name__)


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    logger.info("{0} took too long".format(fn_name))
    # logger.info('Interrupting .')
    thread.interrupt_main()
    # logger.info('Interrupted .')


def exit_after(s):
    """
    use as decorator to exit process if
    function takes longer than s seconds
    """

    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer
