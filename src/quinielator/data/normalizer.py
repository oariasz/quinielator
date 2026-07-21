"""Reglas explícitas para nombres y códigos históricos."""

from __future__ import annotations

import unicodedata


class TeamNameNormalizer:
    """Normaliza variantes sin fusionar sucesores históricos por defecto."""

    _NAME_ALIASES = {
        "IR Iran": "Iran",
        "Korea Republic": "South Korea",
        "Korea DPR": "North Korea",
        "Türkiye": "Turkey",
        "Côte d'Ivoire": "Ivory Coast",
        "USA": "United States",
    }
    _CODE_ALIASES = {
        "GER": "DEU",
        "FRG": "DEU",
        "ROM": "ROU",
        "NED": "NLD",
        "SUI": "CHE",
        "DEN": "DNK",
        "CRC": "CRI",
        "RSA": "ZAF",
        "POR": "PRT",
        "PAR": "PRY",
        "URU": "URY",
        "GRE": "GRC",
        "CRO": "HRV",
        "SLO": "SVN",
        "ALG": "DZA",
        "TUN": "TUN",
        "MAR": "MAR",
        "KSA": "SAU",
        "UAE": "ARE",
        "IRN": "IRN",
        "KOR": "KOR",
        "PRK": "PRK",
        "CIV": "CIV",
        "CMR": "CMR",
        "SEN": "SEN",
        "GHA": "GHA",
        "NGA": "NGA",
        "ANG": "AGO",
        "TOG": "TGO",
        "COD": "COD",
        "ZAI": "COD",
        "CZE": "CZE",
        "TCH": "CSK",
        "URS": "SUN",
        "YUG": "YUG",
        "SCG": "SCG",
    }

    def normalize_name(self, name: str) -> str:
        """Devuelve un nombre canónico conservador."""

        clean = " ".join(name.strip().split())
        return self._NAME_ALIASES.get(clean, clean)

    def normalize_code(self, code: str) -> str:
        """Convierte códigos FIFA frecuentes al código usado por la base mundialista."""

        clean = code.strip().upper()
        return self._CODE_ALIASES.get(clean, clean)

    def key(self, name: str) -> str:
        """Crea una clave tolerante a acentos para emparejamientos secundarios."""

        normalized = unicodedata.normalize("NFKD", self.normalize_name(name))
        ascii_text = "".join(
            character for character in normalized if not unicodedata.combining(character)
        )
        return "".join(character for character in ascii_text.lower() if character.isalnum())
