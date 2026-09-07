#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


def optional_override(funcobj):
    """A decorator indicating primitive methods likely to be overriden when creating a new task problem class.

    Many tasks mixins are defining abstract methods to be implemented. To reduce the burden when implementing
    a new tasks problem class, some other primitive methods have a default implementation.
    Other methods of the mixins are then derived from these primitive methods.

    This decorator helps the problem developper to identify such primitive methods with default implementation that he
    could be interested in overriding.

    Note: This does not modify the function behaviour, and is merely used as documentation.

    """
    funcobj.__islikelytobeoverriden__ = True
    return funcobj
