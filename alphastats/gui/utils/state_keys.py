"""Keys functions for the session state."""

from alphastats.dataset.keys import ConstantsClass
from alphastats.gui.utils.preprocessing_helper import PREPROCESSING_STEPS
from alphastats.llm.uniprot_utils import ExtractedUniprotFields


class StateKeys(metaclass=ConstantsClass):
    """Keys for accessing the session state."""

    USER_SESSION_ID = "user_session_id"
    DATASET = "dataset"
    ORGANISM = "organism"  # TODO: this is essentially a constant

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
    MAX_TOKENS = "max_tokens"
    PROMPT_EXPERIMENTAL_DESIGN = "prompt_experimental_design"
    PROMPT_PROTEIN_DATA = "prompt_protein_data"
    PROMPT_INSTRUCTIONS = "prompt_instructions"


class LLMKeys(metaclass=ConstantsClass):
    """Keys for accessing the session state for LLM."""

    LLM_INTEGRATION = "llm_integration"
    SELECTED_GENES_UP = "selected_genes_up"
    SELECTED_GENES_DOWN = "selected_genes_down"
    RECENT_CHAT_WARNINGS = "recent_chat_warnings"

    IS_INITIALIZED = "is_initialized"
    # Mirrored by StateKeys for handling reactivity and making it available to functions reading from the session state
    SELECTED_UNIPROT_FIELDS = "selected_uniprot_fields"
    INCLUDE_UNIPROT_INTO_INITIAL_PROMPT = "include_uniprot"
    MODEL_NAME = "model_name"
    MAX_TOKENS = "max_tokens"
    PROMPT_EXPERIMENTAL_DESIGN = "prompt_experimental_design"
    PROMPT_PROTEIN_DATA = "prompt_protein_data"
    PROMPT_INSTRUCTIONS = "prompt_instructions"


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
