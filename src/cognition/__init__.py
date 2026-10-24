"""Cognitive and contextual-discovery capabilities for the local agent."""

from importlib import import_module
from typing import Any


__all__ = [
    "CognitiveLearnerSkill",
    "PersonalVocabularyBuilder",
    "SemanticWikiDiscovererSkill",
    "SQLiteCognitiveMemory",
]


def __getattr__(name: str) -> Any:
    modules = {
        "CognitiveLearnerSkill": ".cognitive_learner",
        "PersonalVocabularyBuilder": ".vocabulary_builder",
        "SQLiteCognitiveMemory": ".cognitive_learner",
        "SemanticWikiDiscovererSkill": ".semantic_wiki_discoverer",
    }
    try:
        module_name = modules[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    return getattr(import_module(module_name, __name__), name)