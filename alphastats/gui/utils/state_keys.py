"""Keys functions for the session state."""

from __future__ import annotations

import os
from typing import NamedTuple

from alphastats.dataset.keys import ConstantsClass
from alphastats.gui.utils.preprocessing_helper import PREPROCESSING_STEPS
from alphastats.llm.uniprot_utils import ExtractedUniprotFields


class StateKeys(metaclass=ConstantsClass):
    """Keys for accessing the session state."""

    USER_SESSION_ID = "user_session_id"
    DATASET = "dataset"

    WORKFLOW = "workflow"

    SAVED_ANALYSES = "saved_analyses"

    # LLM
    OPENAI_API_KEY = "openai_api_key"  # pragma: allowlist secret

    LLM_CHATS = "llm_chats"

    ANNOTATION_STORE = "annotation_store"
    SELECTED_ANALYSIS = "selected_analysis"

    # Mirrored by LLMKeys where they are stored in a chat specific manner
    SELECTED_UNIPROT_FIELDS = "selected_uniprot_fields"
    INCLUDE_UNIPROT_INTO_INITIAL_PROMPT = "include_uniprot"
    MODEL_NAME = "model_name"
    BASE_URL = "base_url"
    MAX_TOKENS = "max_tokens"
    PROMPT_EXPERIMENTAL_DESIGN = "prompt_experimental_design"
    PROMPT_PROTEIN_DATA = "prompt_protein_data"
    PROMPT_INSTRUCTIONS = "prompt_instructions"
    ENRICHMENT_COLUMNS = "enrichment_columns"


class LLMKeys(metaclass=ConstantsClass):
    """Keys for accessing the session state for LLM."""

    LLM_INTEGRATION = "llm_integration"
    SELECTED_FEATURES_UP = "selected_features_up"
    SELECTED_FEATURES_DOWN = "selected_features_down"
    RECENT_CHAT_WARNINGS = "recent_chat_warnings"
    ENRICHMENT_ANALYSIS = "enrichment_analysis"

    IS_INITIALIZED = "is_initialized"
    # Mirrored by StateKeys for handling reactivity and making it available to functions reading from the session state
    SELECTED_UNIPROT_FIELDS = "selected_uniprot_fields"
    INCLUDE_UNIPROT_INTO_INITIAL_PROMPT = "include_uniprot"
    MODEL_NAME = "model_name"
    BASE_URL = "base_url"
    MAX_TOKENS = "max_tokens"
    PROMPT_EXPERIMENTAL_DESIGN = "prompt_experimental_design"
    PROMPT_PROTEIN_DATA = "prompt_protein_data"
    PROMPT_INSTRUCTIONS = "prompt_instructions"
    ENRICHMENT_COLUMNS = "enrichment_columns"


class KeySyncNames(metaclass=ConstantsClass):
    """Names used in named tuples for syncing between Key classes."""

    STATE = "StateKey"
    LLM = "LLMKey"
    GET_DEFAULT = "get_default"


SyncedLLMKey = NamedTuple(
    "SyncedLLMKey",
    [
        (KeySyncNames.STATE, str),
        (KeySyncNames.LLM, str),
        (KeySyncNames.GET_DEFAULT, object),
    ],
)

# These keys are synced between the StateKeys and LLMKeys classes.
# They are used upon loading/initializing an LLM chat and on change to any of the widgets to sync the state and chat.
WIDGET_SYNCED_LLM_KEYS: list[SyncedLLMKey] = [
    SyncedLLMKey(
        StateKeys.INCLUDE_UNIPROT_INTO_INITIAL_PROMPT,
        LLMKeys.INCLUDE_UNIPROT_INTO_INITIAL_PROMPT,
        False,  # noqa: FBT003
    ),
    SyncedLLMKey(
        StateKeys.PROMPT_EXPERIMENTAL_DESIGN, LLMKeys.PROMPT_EXPERIMENTAL_DESIGN, None
    ),
    SyncedLLMKey(StateKeys.PROMPT_PROTEIN_DATA, LLMKeys.PROMPT_PROTEIN_DATA, None),
    SyncedLLMKey(StateKeys.PROMPT_INSTRUCTIONS, LLMKeys.PROMPT_INSTRUCTIONS, None),
    SyncedLLMKey(StateKeys.ENRICHMENT_COLUMNS, LLMKeys.ENRICHMENT_COLUMNS, None),
]

# These keys are synced between the StateKeys and LLMKeys classes, but only if the LLM is already initialized with a specific model.
MODEL_SYNCED_LLM_KEYS: list[SyncedLLMKey] = [
    SyncedLLMKey(StateKeys.MODEL_NAME, LLMKeys.MODEL_NAME, None),
    SyncedLLMKey(
        StateKeys.BASE_URL,
        LLMKeys.BASE_URL,
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ),
    SyncedLLMKey(StateKeys.MAX_TOKENS, LLMKeys.MAX_TOKENS, 10000),
]


class SavedAnalysisKeys(metaclass=ConstantsClass):
    """Keys for saved analyses in session state."""

    RESULT = "result"
    METHOD = "method"
    PARAMETERS = "parameters"
    NUMBER = "number"


class DefaultStates(metaclass=ConstantsClass):
    """Default values for some UI components."""

    SELECTED_UNIPROT_FIELDS = [
        ExtractedUniprotFields.NAME,
        ExtractedUniprotFields.GENE,
        ExtractedUniprotFields.FUNCTIONCOMM,
    ]
    WORKFLOW = [
        PREPROCESSING_STEPS.REMOVE_CONTAMINATIONS,
        PREPROCESSING_STEPS.SUBSET,
        PREPROCESSING_STEPS.REPLACE_ZEROES,
        PREPROCESSING_STEPS.LOG2_TRANSFORM,
        PREPROCESSING_STEPS.DROP_UNMEASURED_FEATURES,
    ]
