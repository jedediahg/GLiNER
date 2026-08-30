import pytest

from gliner import GLiNER
from gliner.model import BaseEncoderGLiNER, UniEncoderSpanRelexGLiNER, _entity_types_for_chunk


class _WordsSplitter:
    def __call__(self, text):
        cursor = 0
        for token in text.split():
            start = text.index(token, cursor)
            end = start + len(token)
            cursor = end
            yield token, start, end


def _minimal_encoder_model(cls=BaseEncoderGLiNER):
    model = cls.__new__(cls)
    model.data_processor = type("Processor", (), {"words_splitter": _WordsSplitter()})()
    return model


def test_span_model():
    model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")

    text = """
    Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
    """

    labels = ["person", "award", "date", "competitions", "teams", "person"]

    entities = model.predict_entities(text, labels)

    assert len(entities) > 0


def test_prepare_batch_aligns_per_text_labels_after_empty_text_filtering():
    model = _minimal_encoder_model()

    prepared = model.prepare_batch(
        ["", "John works at Acme"],
        [["wrong"], ["person", "organization"]],
    )

    assert prepared["valid_texts"] == ["John works at Acme"]
    assert prepared["valid_to_orig_idx"] == [1]
    assert prepared["entity_types"] == [["person", "organization"]]


def test_prepare_batch_validates_per_text_label_count():
    model = _minimal_encoder_model()

    with pytest.raises(ValueError, match="Per-text labels must have length 2"):
        model.prepare_batch(["one", "two"], [["label"]])


def test_relex_prepare_batch_aligns_per_text_relations_after_empty_text_filtering():
    model = _minimal_encoder_model(UniEncoderSpanRelexGLiNER)

    prepared = model.prepare_batch(
        ["", "John works at Acme"],
        [["wrong"], ["person", "organization"]],
        relations=[["wrong_relation"], ["works_at"]],
    )

    assert prepared["valid_texts"] == ["John works at Acme"]
    assert prepared["entity_types"] == [["person", "organization"]]
    assert prepared["relation_types"] == [["works_at"]]


def test_relex_prepare_batch_validates_per_text_relation_count():
    model = _minimal_encoder_model(UniEncoderSpanRelexGLiNER)

    with pytest.raises(ValueError, match="Per-text relations must have length 2"):
        model.prepare_batch(
            ["one", "two"],
            [["person"], ["organization"]],
            relations=[["works_at"]],
        )


def test_entity_types_for_chunk_slices_per_row_label_sets():
    """Per-row label sets must follow their rows into each DataLoader chunk.

    The loader hands collate_fn one chunk at a time; without this the decoder indexes the
    whole-batch list with a chunk-local index and rows past the first chunk are decoded against
    another row's labels.
    """
    per_row = [["a"], ["b"], ["c"], ["d"]]
    assert _entity_types_for_chunk(per_row, [2, 3]) == [["c"], ["d"]]
    assert _entity_types_for_chunk(per_row, [0, 1]) == [["a"], ["b"]]


def test_entity_types_for_chunk_passes_shared_label_list_through():
    """A single flat list is shared by every row and must NOT be sliced."""
    shared = ["person", "organization", "location"]
    assert _entity_types_for_chunk(shared, [1, 2]) == shared
    assert _entity_types_for_chunk([], [0]) == []
