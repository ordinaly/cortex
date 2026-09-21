from __future__ import annotations

from cortex.wikidata import statements_from_entity


def test_wikidata_adapter_preserves_multiple_values_qualifiers_rank_and_refs():
    entity = {
        "id": "QTEST",
        "claims": {
            "P36": [
                {
                    "id": "QTEST-A",
                    "rank": "preferred",
                    "mainsnak": {
                        "property": "P36",
                        "snaktype": "value",
                        "datavalue": {
                            "type": "wikibase-entityid",
                            "value": {"id": "QCITY1"},
                        },
                    },
                    "qualifiers": {
                        "P580": [
                            {
                                "property": "P580",
                                "snaktype": "value",
                                "datavalue": {
                                    "type": "time",
                                    "value": {
                                        "time": "+2020-01-01T00:00:00Z",
                                        "precision": 11,
                                        "calendarmodel": "Q1985727",
                                    },
                                },
                            }
                        ]
                    },
                    "references": [{"hash": "ref-a"}],
                },
                {
                    "id": "QTEST-B",
                    "rank": "normal",
                    "mainsnak": {
                        "property": "P36",
                        "snaktype": "value",
                        "datavalue": {
                            "type": "wikibase-entityid",
                            "value": {"id": "QCITY2"},
                        },
                    },
                },
            ]
        },
    }

    rows = list(statements_from_entity(entity))
    assert len(rows) == 2
    assert {row.value for row in rows} == {"wd:QCITY1", "wd:QCITY2"}
    assert rows[0].subject == "wd:QTEST"
    assert rows[0].property == "wd:P36"
    assert rows[0].rank == "preferred"
    assert rows[0].references == ("ref-a",)
    assert rows[0].qualifiers[0].property == "wd:P580"
    assert rows[0].qualifiers[0].value.startswith("wdlit:time:")


def test_deprecated_claims_are_skipped_by_default():
    entity = {
        "id": "QTEST",
        "claims": {
            "P1": [
                {
                    "id": "old",
                    "rank": "deprecated",
                    "mainsnak": {
                        "property": "P1",
                        "snaktype": "value",
                        "datavalue": {"type": "string", "value": "old"},
                    },
                }
            ]
        },
    }
    assert list(statements_from_entity(entity)) == []
    assert len(list(statements_from_entity(entity, include_deprecated=True))) == 1
