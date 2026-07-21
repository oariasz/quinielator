import pytest

from quinielator.domain import MatchSign, Stage


def test_stage_aliases_include_spanish_and_typo() -> None:
    assert Stage.parse("final") is Stage.FINAL
    assert Stage.parse("semis") is Stage.SEMIFINAL
    assert Stage.parse("seminfinal") is Stage.SEMIFINAL
    assert Stage.parse("cuartos") is Stage.QUARTERFINAL


def test_unknown_stage_is_explicit() -> None:
    assert Stage.parse("ronda inventada") is Stage.OTHER


def test_match_sign_maps_outcome_to_one_x_two() -> None:
    assert MatchSign.from_outcome("H") is MatchSign.HOME
    assert MatchSign.from_outcome("D") is MatchSign.DRAW
    assert MatchSign.from_outcome("A") is MatchSign.AWAY
    with pytest.raises(ValueError, match="Resultado desconocido"):
        MatchSign.from_outcome("?")
