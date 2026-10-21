from niyam.cbic import Circular
from niyam.ingest import Chunk
from niyam.search import Hit, circular_ranking, reciprocal_rank_fusion


def row(**overrides):
    base = {"id": 1, "circularNo": "164/20/2021-GST", "circularDt": "2021-10-06T05:30:00+05:30",
            "circularName": "Clariﬁcation  on rates", "circularCategory": "Circulars CGST",
            "docFileName": "c.pdf", "docFilePathHi": ""}
    return base | overrides


def test_circular_from_api_normalises_fields():
    c = Circular.from_api(row())
    assert (c.number, c.date, c.subject, c.has_hindi) == ("164/20/2021-GST", "2021-10-06", "Clarification on rates", False)


def test_corrigenda_get_unique_labels():
    assert Circular.from_api(row(circularNo="Corrigendum")).number == "Corrigendum dated 2021-10-06"


def hit(chunk_id: str, number: str) -> Hit:
    return Hit(Chunk(chunk_id, 1, number, "2024-01-01", "s", 1, 1, "t"), 0.0)


def test_rrf_rewards_agreement_between_rankings():
    a, b, c = hit("a", "1"), hit("b", "2"), hit("c", "3")
    fused = reciprocal_rank_fusion([a, b], [c, b])
    assert fused[0].chunk.chunk_id == "b"  # second in both lists beats first in one


def test_circular_ranking_dedupes_in_order():
    hits = [hit("x-0", "10"), hit("y-0", "20"), hit("x-1", "10"), hit("z-0", "30")]
    assert circular_ranking(hits) == ["10", "20", "30"]
