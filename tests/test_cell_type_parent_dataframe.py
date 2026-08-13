import pandas as pd

from scprint2.tasks import cell_emb


def _records():
    return pd.DataFrame(
        {
            "ontology_id": ["CL:1", "CL:2"],
            "parents__ontology_id": [{"CL:9", "CL:8"}, {None}],
        }
    )


def test_cell_type_parents_support_current_lamin_query_api(monkeypatch):
    class CurrentQuery:
        def all(self):
            return self

        def df(self, include):
            assert include == ["parents__ontology_id", "ontology_id"]
            return _records()

    monkeypatch.setattr(cell_emb.bt.CellType, "filter", lambda: CurrentQuery())

    result = cell_emb._cell_type_parent_dataframe()

    assert result.loc["CL:1", "parents__ontology_id"] == "CL:8, CL:9"
    assert result.loc["CL:2", "parents__ontology_id"] == ""


def test_cell_type_parents_disable_limit_on_new_lamin_query_api(monkeypatch):
    class NewQuery:
        def to_dataframe(self, *, include, limit):
            assert include == ["parents__ontology_id", "ontology_id"]
            assert limit is None
            return _records()

    monkeypatch.setattr(cell_emb.bt.CellType, "filter", lambda: NewQuery())

    result = cell_emb._cell_type_parent_dataframe()

    assert len(result) == 2
