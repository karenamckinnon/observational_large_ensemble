#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `observational_large_ensemble` package."""


import unittest
from click.testing import CliRunner

from observational_large_ensemble import observational_large_ensemble
from observational_large_ensemble import cli


class TestObservational_large_ensemble(unittest.TestCase):
    """Tests for `observational_large_ensemble` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'observational_large_ensemble.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
