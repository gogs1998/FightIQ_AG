import pytest
import os
import sys
import json
import pandas as pd

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qa_gate import run_qa_gate

def test_qa_gate(capsys):
    """
    Runs the QA Gate script and asserts it exits with code 0.
    Captures stdout to check for warnings if needed.
    """
    try:
        run_qa_gate()
    except SystemExit as e:
        assert e.code == 0, "QA Gate failed (non-zero exit code)"
        
    # Optional: Check output for specific critical warnings if we wanted to enforce strictness
    captured = capsys.readouterr()
    assert "CRITICAL" not in captured.out
    assert "QA FAILED" not in captured.out
