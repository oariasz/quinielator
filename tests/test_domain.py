from quinielator.domain import Stage


def test_stage_aliases_include_spanish_and_typo() -> None:
    assert Stage.parse("final") is Stage.FINAL
    assert Stage.parse("semis") is Stage.SEMIFINAL
    assert Stage.parse("seminfinal") is Stage.SEMIFINAL
    assert Stage.parse("cuartos") is Stage.QUARTERFINAL


def test_unknown_stage_is_explicit() -> None:
    assert Stage.parse("ronda inventada") is Stage.OTHER
