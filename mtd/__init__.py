# mtd/__init__.py

from .task1_vulnerability_classification import run as classify
from .task2_line_localization            import run as localize
from .task3_syntax_risk_prediction       import run as predict_syntax
from .task4_dependency_propagation_risk  import run as propagation_risk

__all__ = ["classify", "localize", "predict_syntax", "propagation_risk"]