from src.agents.classifier import TaskComplexity, classify_task


def test_simple_equipment_lookup_bypasses_planning():
    result = classify_task("Tell me about P-101. What is the equipment?")
    assert result.complexity is TaskComplexity.SIMPLE_LOOKUP
    assert result.requires_planning is False
    assert result.equipment_id == "P-101"


def test_what_is_phrasing_also_classified_as_simple_lookup():
    result = classify_task("What is V-101?")
    assert result.requires_planning is False
    assert result.equipment_id == "V-101"


def test_comparison_task_requires_full_planning():
    result = classify_task(
        "Analyze V-101: compare observed pressure 120 bar against approved "
        "limit 100 bar, review history, retrieve SOP, and recommend."
    )
    assert result.requires_planning is True
    assert result.equipment_id == "V-101"
    assert result.complexity in (
        TaskComplexity.MULTI_STEP_INVESTIGATION, TaskComplexity.INSPECTION_ANALYSIS,
    )


def test_investigation_task_classified_as_multi_step():
    result = classify_task(
        "Investigate whether P-101 should be taken out of service."
    )
    assert result.requires_planning is True
    assert result.complexity is TaskComplexity.MULTI_STEP_INVESTIGATION


def test_task_without_equipment_id_defaults_to_full_planning():
    result = classify_task("Summarise the attached readings.")
    assert result.requires_planning is True
    assert result.equipment_id is None


def test_ambiguous_equipment_mention_without_lookup_phrase_still_plans():
    # No clear "tell me about" / "what is" phrasing -- must not guess a
    # fast path for an ambiguous request just because an id is present.
    result = classify_task("P-101 readings from last week")
    assert result.requires_planning is True
