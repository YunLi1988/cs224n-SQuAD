#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common definitions for QA
"""

from util import one_hot

LBLS = [
    "ANS",
    "O",
    "EMPTY"
    ]
NONE = "EMPTY"
LMAP = {k: one_hot(5,i) for i, k in enumerate(LBLS)}
NUM = "NNNUMMM"
UNK = "UUUNKKK"

EMBED_SIZE = 100
