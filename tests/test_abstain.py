"""Tests for shared abstention-phrase detection."""

import pytest

from utils.abstain import ABSTAIN_PHRASES, is_abstain

# ---------------------------------------------------------------------------
# is_abstain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Insufficient evidence to determine the answer.",
        "I do not know based on the provided context.",
        "I don't know.",
        "The answer cannot be determined from the filing.",
        "The context does not contain the requested figure.",
        "The context does not mention 2022 CAPEX.",
        "No information about FY2023 revenue is provided.",
        "Unable to find relevant information in the filing.",
    ],
)
def test_is_abstain_recognizes_phrase(text):
    assert is_abstain(text), f"expected abstain detection on: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "Apple's FY2022 revenue was $394 billion.",
        "Net income totaled $99.8B according to the income statement.",
        "The CAPEX increased by 12% year over year.",
    ],
)
def test_is_abstain_rejects_substantive_answer(text):
    assert not is_abstain(text)


def test_is_abstain_case_insensitive():
    assert is_abstain("INSUFFICIENT EVIDENCE")
    assert is_abstain("I Do Not Know")


def test_is_abstain_handles_none_and_empty():
    assert is_abstain(None) is False
    assert is_abstain("") is False


def test_abstain_phrases_includes_hypo_native_marker():
    # Prehop's pipeline-specific abstain phrase must remain in the list;
    # removing it would silently mis-classify Hypo's own abstentions as
    # Incorrect Answer with answer_attempted=1.
    assert "insufficient evidence" in ABSTAIN_PHRASES
