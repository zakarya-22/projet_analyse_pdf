import logging
from datetime import datetime
from pathlib import Path


def _init_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("credit_audit")
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger


class AuditReporter:
    def __init__(self, log_path: str | Path = "logs/audit.log"):
        self._logger = _init_logger(Path(log_path))

    def compile(
        self,
        client_id: int | str,
        prob: float,
        top_feature: str,
        top_value: float,
        regulatory_text: str,
    ) -> dict:
        report = {
            "client_id": client_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "probability": prob,
            "top_feature": top_feature,
            "top_value": top_value,
            "regulatory_analysis": regulatory_text,
        }
        self._logger.info(
            f"Client {client_id} | PD={prob:.2%} | "
            f"Feature={top_feature}({top_value:.4f}) | RAG declenche"
        )
        return report

    def log_batch(self, n_total: int, n_high_risk: int, threshold: float) -> None:
        self._logger.info(
            f"Batch | {n_total} clients analyses | "
            f"{n_high_risk} a risque eleve (seuil={threshold:.0%})"
        )

    def to_markdown(self, report: dict) -> str:
        return (
            f"# Rapport d'Audit - Risque de Credit\n\n"
            f"**Client ID :** {report['client_id']}  \n"
            f"**Date :** {report['timestamp']}  \n\n"
            f"---\n\n"
            f"## Score de Risque\n\n"
            f"| Indicateur | Valeur |\n"
            f"|---|---|\n"
            f"| Probabilite de defaut | **{report['probability']:.2%}** |\n"
            f"| Facteur declencheur | `{report['top_feature']}` = {report['top_value']:.4f} |\n\n"
            f"## Analyse Reglementaire\n\n"
            f"{report['regulatory_analysis']}\n\n"
            f"---\n"
            f"*Rapport genere localement - Pipeline RGPD-compliant*\n"
        )
