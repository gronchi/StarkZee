"""
conftest.py — shared pytest fixtures and path setup for starkzee tests.
"""
import sys
import os
import pytest

# Make sure starkzee is importable from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
