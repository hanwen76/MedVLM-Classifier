from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ImageWikiQARetriever:
    """Local knowledge retriever triggered after disease classification.

    Knowledge base format (JSON):
    {
      "湿疹": {"guideline": "...", "references": ["..."]},
      "银屑病": {...}
    }
    """

    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)
        with self.knowledge_base_path.open("r", encoding="utf-8") as f:
            self.db: dict[str, Any] = json.load(f)

    def retrieve(self, disease_name: str) -> dict[str, Any]:
        if disease_name in self.db:
            return {"disease": disease_name, "result": self.db[disease_name], "found": True}

        # Fallback fuzzy contains matching for class label aliases.
        disease_name_l = disease_name.lower()
        for key, value in self.db.items():
            if disease_name_l in key.lower() or key.lower() in disease_name_l:
                return {"disease": key, "result": value, "found": True}

        return {
            "disease": disease_name,
            "result": {"guideline": "No local guideline found.", "references": []},
            "found": False,
        }
