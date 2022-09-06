#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import disable_caching


def pytest_configure():
    # globally disable caching with datasets
    disable_caching()
