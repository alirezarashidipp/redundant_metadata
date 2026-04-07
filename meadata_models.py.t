# schemas.py

from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from config_loader import get_prompts


class ActionCategory(str, Enum):
    MIGRATION     = "migration"
    DEBUG         = "debug"
    ENHANCE       = "enhance"
    REFACTOR      = "refactor"
    TESTING       = "testing"
    DEPLOYMENT    = "deployment"
    RESEARCH      = "research"
    OTHER         = "other"
    DOCUMENTATION = "documentation"
    NOT_FOUND     = None


class detail_in_primary_intent(str, Enum):
    WITH_DETAILS = "with_details"
    MIDDLE       = "middle"
    NO_DETAILS   = "no_details"


def make_jira_analysis(prompt_version: str):
    """
    Reads prompts.yaml[prompt_version] and returns a JiraAnalysis class
    where every Field description comes from that version.

    Both main_prompt and field descriptions are from the same version.
    """
    s = get_prompts()[prompt_version]  # s = sections

    class Who(BaseModel):
        identified: bool          = Field(default=False, description=s["who"])
        evidence:   Optional[str] = Field(default=None,  description=s["who_evidence"])

    class What(BaseModel):
        identified:                     bool                     = Field(default=False,                          description=s["what"])
        category:                       ActionCategory           = Field(default=ActionCategory.NOT_FOUND,       description=s["what_category"])
        intent_evidence:                Optional[str]            = Field(default=None,                           description=s["what_evidence"])
        the_level_of_details_in_intent: detail_in_primary_intent = Field(default=detail_in_primary_intent.NO_DETAILS, description=s["level_of_details_in_what"])

    class Why(BaseModel):
        identified:     bool          = Field(default=False, description=s["why"])
        value_evidence: Optional[str] = Field(default=None,  description=s["why_evidence"])

    class CustomerImpact(BaseModel):
        identified:      bool          = Field(default=False, description=s["customer_impact"])
        impact_evidence: Optional[str] = Field(default=None,  description=s["impact_evidence"])

    class AgilePrincipleCheck(BaseModel):
        presence_ac:           bool          = Field(default=False, description=s["ac_defined"])
        english_text_feedback: Optional[str] = Field(default=None,  description=s["english_text_feedback"])
        overal_feedback:       str           = Field(description=s["overal_feedback"])

    class JiraAnalysis(BaseModel):
        reasoning:          str       = Field(description=s["reasoning"])
        who:                Who
        what:               What
        why:                Why
        customer_impact:    CustomerImpact
        ac_defined:         AgilePrincipleCheck
        grooming_questions: List[str] = Field(description=s["grooming_questions"])
        agile_standard:     bool      = Field(default=False)

    return JiraAnalysis
