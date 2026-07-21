"""Entidades inmutables que expresan el lenguaje del experimento."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class Outcome(StrEnum):
    """Resultado desde la perspectiva del equipo listado como local."""

    HOME = "H"
    DRAW = "D"
    AWAY = "A"


class Stage(StrEnum):
    """Fases normalizadas que interesan a la experiencia interactiva."""

    GROUP = "group"
    ROUND_OF_32 = "round_of_32"
    ROUND_OF_16 = "round_of_16"
    QUARTERFINAL = "quarterfinal"
    SEMIFINAL = "semifinal"
    THIRD_PLACE = "third_place"
    FINAL = "final"
    OTHER = "other"

    @classmethod
    def parse(cls, value: str) -> Stage:
        """Normaliza nombres en español, inglés y errores frecuentes."""

        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "final": cls.FINAL,
            "semifinal": cls.SEMIFINAL,
            "semifinals": cls.SEMIFINAL,
            "semi_finals": cls.SEMIFINAL,
            "semis": cls.SEMIFINAL,
            "seminfinal": cls.SEMIFINAL,
            "cuartos": cls.QUARTERFINAL,
            "quarterfinal": cls.QUARTERFINAL,
            "quarterfinals": cls.QUARTERFINAL,
            "quarter_finals": cls.QUARTERFINAL,
            "quarter_finals_stage": cls.QUARTERFINAL,
            "group_stage": cls.GROUP,
            "group": cls.GROUP,
            "round_of_32": cls.ROUND_OF_32,
            "round_of_32_stage": cls.ROUND_OF_32,
            "round_of_16": cls.ROUND_OF_16,
            "round_of_16_stage": cls.ROUND_OF_16,
            "third_place": cls.THIRD_PLACE,
            "third_place_match": cls.THIRD_PLACE,
        }
        return aliases.get(normalized, cls.OTHER)


@dataclass(frozen=True)
class Team:
    """Selección con identidad original y normalizada."""

    code: str
    name: str
    original_name: str
    confederation: str = "UNKNOWN"


@dataclass(frozen=True)
class Tournament:
    """Edición de la Copa Mundial."""

    tournament_id: str
    year: int
    name: str
    host_country: str
    has_final: bool


@dataclass(frozen=True)
class Match:
    """Partido histórico con tiempos y penaltis separados."""

    match_id: str
    tournament_id: str
    tournament_year: int
    played_on: date
    stage: Stage
    stage_name: str
    home_team: Team
    away_team: Team
    home_goals_90: int
    away_goals_90: int
    home_goals_total: int
    away_goals_total: int
    home_penalties: int = 0
    away_penalties: int = 0
    extra_time: bool = False
    penalty_shootout: bool = False
    host_country: str = ""

    @property
    def outcome_90(self) -> Outcome:
        """Resultado 1X2 al final del tiempo reglamentario."""

        if self.home_goals_90 > self.away_goals_90:
            return Outcome.HOME
        if self.home_goals_90 < self.away_goals_90:
            return Outcome.AWAY
        return Outcome.DRAW

    @property
    def advancing_team_code(self) -> str | None:
        """Código del equipo que avanzó cuando el partido produjo un ganador."""

        if self.home_goals_total > self.away_goals_total:
            return self.home_team.code
        if self.home_goals_total < self.away_goals_total:
            return self.away_team.code
        if self.home_penalties > self.away_penalties:
            return self.home_team.code
        if self.home_penalties < self.away_penalties:
            return self.away_team.code
        return None


@dataclass(frozen=True)
class FifaRankingEntry:
    """Publicación del ranking FIFA para una selección."""

    team_code: str
    published_on: date
    rank: int
    points: float


@dataclass(frozen=True)
class MatchPrediction:
    """Predicción probabilística producida antes de conocer el resultado."""

    match_id: str
    tournament_year: int
    played_on: date
    stage: Stage
    home_team_code: str
    home_team_name: str
    away_team_code: str
    away_team_name: str
    expected_home_goals: float
    expected_away_goals: float
    predicted_home_goals: int
    predicted_away_goals: int
    probability_home: float
    probability_draw: float
    probability_away: float
    predicted_advancing_team: str | None
    model_name: str
    trained_matches: int
    as_of_date: date

    @property
    def predicted_outcome(self) -> Outcome:
        probabilities = {
            Outcome.HOME: self.probability_home,
            Outcome.DRAW: self.probability_draw,
            Outcome.AWAY: self.probability_away,
        }
        return max(probabilities, key=probabilities.__getitem__)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Resumen de calidad para un conjunto de predicciones."""

    matches: int
    outcome_accuracy: float
    exact_score_accuracy: float
    advancing_accuracy: float | None
    goals_mae: float
    log_loss: float
    brier_score: float
    home_goal_mae: float
    away_goal_mae: float
    extra: dict[str, float] = field(default_factory=dict)
