"""Reporte visual por partido para una edición concreta del Mundial."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd

from quinielator.domain import MatchSign


@dataclass(frozen=True)
class WorldCupReportArtifacts:
    """Rutas generadas para consultar o reutilizar el reporte."""

    csv: Path
    markdown: Path
    html: Path
    svg: Path


class WorldCupVisualReport:
    """Crea una tabla 1/X/2 coloreada evaluada exclusivamente a 90 minutos."""

    STAGE_LABELS = {
        "group": "Grupos",
        "round_of_32": "Ronda de 32",
        "round_of_16": "Octavos",
        "quarterfinal": "Cuartos",
        "semifinal": "Semifinal",
        "third_place": "Tercer puesto",
        "final": "Final",
        "other": "Otra fase",
    }

    def build_table(self, predictions: pd.DataFrame, year: int) -> pd.DataFrame:
        """Selecciona una edición y añade signos, etiquetas y estados visuales."""

        selected = predictions[predictions["tournament_year"].eq(year)].copy()
        if selected.empty:
            raise ValueError(f"No hay predicciones para el Mundial {year}")
        selected = selected.sort_values(["match_date", "match_id"]).reset_index(drop=True)
        selected["predicted_sign"] = selected["predicted_outcome"].map(
            lambda value: MatchSign.from_outcome(str(value)).value
        )
        selected["actual_sign"] = selected["actual_outcome"].map(
            lambda value: MatchSign.from_outcome(str(value)).value
        )
        selected["sign_correct"] = selected["predicted_sign"].eq(selected["actual_sign"])
        selected["exact_score"] = selected["predicted_home_goals"].eq(
            selected["actual_home_goals"]
        ) & selected["predicted_away_goals"].eq(selected["actual_away_goals"])
        selected["number"] = range(1, len(selected) + 1)
        selected["stage_label"] = (
            selected["stage"].map(self.STAGE_LABELS).fillna(selected["stage_name"])
        )
        selected["match"] = (
            selected["home_team_name"].astype(str) + " — " + selected["away_team_name"].astype(str)
        )
        selected["predicted_score"] = (
            selected["predicted_home_goals"].astype(int).astype(str)
            + "–"
            + selected["predicted_away_goals"].astype(int).astype(str)
        )
        selected["actual_score_90"] = (
            selected["actual_home_goals"].astype(int).astype(str)
            + "–"
            + selected["actual_away_goals"].astype(int).astype(str)
        )
        selected["predicted_result"] = selected.apply(
            lambda row: self._result_label(row, "predicted_sign"), axis=1
        )
        selected["actual_result_90"] = selected.apply(
            lambda row: self._result_label(row, "actual_sign"), axis=1
        )
        selected["after_90_note"] = selected.apply(self._after_90_note, axis=1)
        selected["evaluation"] = selected.apply(self._evaluation_label, axis=1)
        return selected[
            [
                "number",
                "match_id",
                "match_date",
                "stage_label",
                "home_team_name",
                "away_team_name",
                "match",
                "predicted_score",
                "predicted_sign",
                "predicted_result",
                "probability_home",
                "probability_draw",
                "probability_away",
                "actual_score_90",
                "actual_sign",
                "actual_result_90",
                "sign_correct",
                "exact_score",
                "after_90_note",
                "evaluation",
            ]
        ]

    def write(
        self,
        predictions: pd.DataFrame,
        *,
        year: int,
        model_name: str,
        output_directory: Path,
    ) -> WorldCupReportArtifacts:
        """Guarda CSV, Markdown, HTML y resumen SVG sin depender de un navegador."""

        output_directory.mkdir(parents=True, exist_ok=True)
        table = self.build_table(predictions, year)
        stem = f"mundial_{year}_{model_name}"
        csv_path = output_directory / f"{stem}.csv"
        markdown_path = output_directory / f"{stem}.md"
        html_path = output_directory / f"{stem}.html"
        svg_path = output_directory / f"{stem}_resumen.svg"
        table.to_csv(csv_path, index=False)
        markdown_path.write_text(self._markdown(table, year, model_name), encoding="utf-8")
        html_path.write_text(self._html(table, year, model_name), encoding="utf-8")
        svg_path.write_text(self._svg(table, year, model_name), encoding="utf-8")
        return WorldCupReportArtifacts(csv_path, markdown_path, html_path, svg_path)

    @staticmethod
    def _result_label(row: pd.Series, sign_column: str) -> str:
        sign = str(row[sign_column])
        if sign == MatchSign.HOME.value:
            return f"Gana {row['home_team_name']}"
        if sign == MatchSign.AWAY.value:
            return f"Gana {row['away_team_name']}"
        return "Empate"

    @staticmethod
    def _after_90_note(row: pd.Series) -> str:
        if int(row.get("penalty_shootout", 0)):
            return "Hubo penaltis después; no cuentan para el signo"
        if int(row["actual_home_goals_total"]) != int(row["actual_home_goals"]) or int(
            row["actual_away_goals_total"]
        ) != int(row["actual_away_goals"]):
            return "Hubo prórroga después; no cambia el signo a 90′"
        return ""

    @staticmethod
    def _evaluation_label(row: pd.Series) -> str:
        if bool(row["exact_score"]) and bool(row["sign_correct"]):
            return "Signo y marcador exacto"
        if bool(row["exact_score"]):
            return "Marcador exacto; signo fallado"
        if bool(row["sign_correct"]):
            return "Signo acertado"
        return "Signo fallado"

    @staticmethod
    def _summary(table: pd.DataFrame) -> dict[str, float | int]:
        matches = len(table)
        hits = int(table["sign_correct"].sum())
        exact = int(table["exact_score"].sum())
        penalties = int(table["after_90_note"].str.contains("penaltis").sum())
        return {
            "matches": matches,
            "hits": hits,
            "misses": matches - hits,
            "accuracy": hits / matches,
            "exact": exact,
            "exact_accuracy": exact / matches,
            "penalties": penalties,
        }

    def _markdown(self, table: pd.DataFrame, year: int, model_name: str) -> str:
        summary = self._summary(table)
        lines = [
            f"# Predicciones completas del Mundial {year}",
            "",
            f"Modelo: `{model_name}`. Todos los signos y marcadores se evalúan a 90 minutos.",
            "Las prórrogas y tandas de penaltis posteriores se muestran como nota, "
            "pero no cuentan.",
            "",
            f"- Aciertos de signo: **{summary['hits']}/{summary['matches']} "
            f"({summary['accuracy']:.1%})**",
            f"- Marcadores exactos: **{summary['exact']}/{summary['matches']} "
            f"({summary['exact_accuracy']:.1%})**",
            f"- Partidos con tanda posterior ignorada: **{summary['penalties']}**",
            "- Leyenda: 🟩 signo acertado · 🟥 signo fallado · 🎯 marcador exacto",
            "",
            "| # | Fecha | Fase | Partido | Predicción | Signo P | Real 90′ | "
            "Signo R | Evaluación |",
            "|---:|---|---|---|---:|:---:|---:|:---:|---|",
        ]
        for _, row in table.iterrows():
            marker = "🟩" if bool(row["sign_correct"]) else "🟥"
            if bool(row["exact_score"]):
                marker += " 🎯"
            note = str(row["after_90_note"])
            evaluation = marker if not note else f"{marker} · {note}"
            values = [
                str(int(row["number"])),
                str(row["match_date"]),
                str(row["stage_label"]),
                str(row["match"]),
                str(row["predicted_score"]),
                str(row["predicted_sign"]),
                str(row["actual_score_90"]),
                str(row["actual_sign"]),
                evaluation,
            ]
            lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
        return "\n".join(lines) + "\n"

    def _html(self, table: pd.DataFrame, year: int, model_name: str) -> str:
        summary = self._summary(table)
        hit_width = float(summary["accuracy"]) * 100.0
        rows: list[str] = []
        for _, row in table.iterrows():
            correct = bool(row["sign_correct"])
            exact = bool(row["exact_score"])
            row_class = (
                "hit exact"
                if correct and exact
                else "hit"
                if correct
                else "miss exact-score"
                if exact
                else "miss"
            )
            status = (
                "🎯 Signo y marcador exactos"
                if correct and exact
                else "✓ Signo acertado"
                if correct
                else "✕ Signo fallado · marcador exacto"
                if exact
                else "✕ Signo fallado"
            )
            note = str(row["after_90_note"])
            note_html = f'<div class="note">{escape(note)}</div>' if note else ""
            rows.append(
                f"""<tr class="{row_class}">
<td>{int(row["number"])}</td><td>{escape(str(row["match_date"]))}</td>
<td>{escape(str(row["stage_label"]))}</td>
<td><strong>{escape(str(row["home_team_name"]))}</strong><br>vs.<br>
<strong>{escape(str(row["away_team_name"]))}</strong></td>
<td class="score">{escape(str(row["predicted_score"]))}</td>
<td><span class="sign">{escape(str(row["predicted_sign"]))}</span><br>
{escape(str(row["predicted_result"]))}</td>
<td class="prob">1: {float(row["probability_home"]):.1%}<br>
X: {float(row["probability_draw"]):.1%}<br>2: {float(row["probability_away"]):.1%}</td>
<td class="score">{escape(str(row["actual_score_90"]))}</td>
<td><span class="sign">{escape(str(row["actual_sign"]))}</span><br>
{escape(str(row["actual_result_90"]))}</td>
<td><strong>{status}</strong>{note_html}</td>
</tr>"""
            )
        return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predicciones Mundial {year} — {escape(model_name)}</title>
<style>
:root{{--green:#16a34a;--green-bg:#ecfdf5;--red:#dc2626;--red-bg:#fef2f2;
--blue:#2563eb;--ink:#172033;--muted:#657084;--line:#dce2ea}}
*{{box-sizing:border-box}} body{{font-family:Inter,system-ui,-apple-system,sans-serif;
margin:0;background:#f4f7fb;color:var(--ink)}} main{{max-width:1500px;margin:auto;padding:32px}}
h1{{margin:0 0 8px}} .subtitle{{color:var(--muted);margin-bottom:24px}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:14px;margin:20px 0}}
.card{{background:white;border:1px solid var(--line);border-radius:14px;padding:18px}}
.value{{font-size:30px;font-weight:800}} .label{{color:var(--muted);font-size:13px}}
.bar{{height:28px;background:var(--red);border-radius:10px;overflow:hidden;margin:18px 0 8px}}
.bar span{{display:block;height:100%;width:{hit_width:.3f}%;background:var(--green)}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:20px;color:var(--muted)}}
.dot{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px}}
.table-wrap{{overflow:auto;background:white;border:1px solid var(--line);border-radius:14px}}
table{{border-collapse:collapse;width:100%;min-width:1180px;font-size:13px}}
th{{position:sticky;top:0;background:#172033;color:white;padding:12px 9px;text-align:left}}
td{{border-bottom:1px solid var(--line);padding:10px 9px;vertical-align:middle}}
tr.hit{{background:var(--green-bg);border-left:6px solid var(--green)}}
tr.miss{{background:var(--red-bg);border-left:6px solid var(--red)}}
tr.exact{{background:#dcfce7}} .score{{font-size:20px;font-weight:800;text-align:center}}
.sign{{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:8px;
background:#172033;color:white;font-size:17px;font-weight:800}} .prob{{white-space:nowrap}}
.note{{font-size:11px;color:var(--muted);margin-top:5px;max-width:180px}}
@media(max-width:800px){{main{{padding:18px}}.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>Mundial {year}: predicción vs. resultado real</h1>
<p class="subtitle">Modelo <strong>{escape(model_name)}</strong> · Signos 1/X/2 y marcadores
evaluados a 90 minutos. Prórroga y penaltis posteriores no modifican el resultado.</p>
<section class="cards">
<div class="card"><div class="value">{summary["hits"]}/{summary["matches"]}</div>
<div class="label">signos acertados</div></div>
<div class="card"><div class="value">{summary["accuracy"]:.1%}</div>
<div class="label">accuracy 1/X/2</div></div>
<div class="card"><div class="value">{summary["exact"]}</div>
<div class="label">marcadores exactos</div></div>
<div class="card"><div class="value">{summary["penalties"]}</div>
<div class="label">tandas posteriores ignoradas</div></div>
</section>
<div class="bar" title="Verde: aciertos; rojo: fallos"><span></span></div>
<div class="legend"><span><i class="dot" style="background:var(--green)"></i>Acierto</span>
<span><i class="dot" style="background:var(--red)"></i>Fallo</span>
<span>🎯 Signo y marcador exacto</span></div>
<div class="table-wrap"><table><thead><tr><th>#</th><th>Fecha</th><th>Fase</th>
<th>Partido</th><th>Pred.</th><th>Signo predicho</th><th>Probabilidades</th>
<th>Real 90′</th><th>Signo real</th><th>Evaluación</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
</main></body></html>"""

    def _svg(self, table: pd.DataFrame, year: int, model_name: str) -> str:
        summary = self._summary(table)
        width, height = 1000, 310
        left, bar_width = 70, 860
        hit_width = bar_width * float(summary["accuracy"])
        actual_counts = table["actual_sign"].value_counts()
        predicted_counts = table["predicted_sign"].value_counts()
        actual_summary = (
            f"1: {int(actual_counts.get('1', 0))} · X: {int(actual_counts.get('X', 0))} · "
            f"2: {int(actual_counts.get('2', 0))}"
        )
        predicted_summary = (
            f"1: {int(predicted_counts.get('1', 0))} · "
            f"X: {int(predicted_counts.get('X', 0))} · "
            f"2: {int(predicted_counts.get('2', 0))}"
        )
        footer = (
            f"Verde = signo acertado · Rojo = signo fallado · Total: {summary['matches']} partidos"
        )
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" rx="18" fill="#f8fafc"/>
<text x="500" y="38" text-anchor="middle" font-family="system-ui" font-size="23"
font-weight="700" fill="#172033">Mundial {year} · signos 1/X/2 · {escape(model_name)}</text>
<text x="500" y="66" text-anchor="middle" font-family="system-ui" font-size="14"
fill="#657084">Resultado reglamentario; prórroga y penaltis no cuentan</text>
<rect x="{left}" y="94" width="{bar_width}" height="48" rx="12" fill="#dc2626"/>
<rect x="{left}" y="94" width="{hit_width:.1f}" height="48" rx="12" fill="#16a34a"/>
<text x="{left + hit_width / 2:.1f}" y="125" text-anchor="middle" font-family="system-ui"
font-size="17" font-weight="700" fill="white">{summary["hits"]} aciertos</text>
<text x="{left + hit_width + (bar_width - hit_width) / 2:.1f}" y="125"
text-anchor="middle" font-family="system-ui" font-size="17" font-weight="700"
fill="white">{summary["misses"]} fallos</text>
<text x="165" y="190" text-anchor="middle" font-family="system-ui" font-size="29"
font-weight="800" fill="#16a34a">{summary["accuracy"]:.1%}</text>
<text x="165" y="214" text-anchor="middle" font-family="system-ui" font-size="13"
fill="#657084">accuracy de signo</text>
<text x="385" y="190" text-anchor="middle" font-family="system-ui" font-size="29"
font-weight="800" fill="#2563eb">{summary["exact"]}</text>
<text x="385" y="214" text-anchor="middle" font-family="system-ui" font-size="13"
fill="#657084">marcadores exactos</text>
<text x="650" y="184" text-anchor="middle" font-family="system-ui" font-size="15"
font-weight="700" fill="#172033">Signos reales</text>
<text x="650" y="210" text-anchor="middle" font-family="system-ui" font-size="14"
fill="#657084">{escape(actual_summary)}</text>
<text x="850" y="184" text-anchor="middle" font-family="system-ui" font-size="15"
font-weight="700" fill="#172033">Signos predichos</text>
<text x="850" y="210" text-anchor="middle" font-family="system-ui" font-size="14"
fill="#657084">{escape(predicted_summary)}</text>
<text x="500" y="272" text-anchor="middle" font-family="system-ui" font-size="13"
fill="#657084">{escape(footer)}</text>
</svg>
"""
