import pytest

from subjectmate.settings import Settings


def test_settings_accept_valid_chunk_configuration():
    Settings(chunk_size=500, chunk_overlap=80).validate()


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (500, -1), (500, 500)],
)
def test_settings_reject_invalid_chunk_configuration(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        Settings(chunk_size=chunk_size, chunk_overlap=chunk_overlap).validate()
