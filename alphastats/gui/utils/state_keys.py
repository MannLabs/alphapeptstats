"""Keys and helper functions for the session state."""

from alphastats.dataset.keys import ConstantsClass
from alphastats.gui.utils.preprocessing_helper import PREPROCESSING_STEPS
from alphastats.llm.uniprot_utils import ExtractedUniprotFields


class StateKeys(metaclass=ConstantsClass):
    """Keys for accessing the session state."""

    USER_SESSION_ID = "user_session_id"
    DATASET = "dataset"

    WORKFLOW = "workflow"

    ANALYSIS_LIST = "analysis_list"

    # LLM
    OPENAI_API_KEY = "openai_api_key"  # pragma: allowlist secret
    MODEL_NAME = "model_name"
    LLM_INPUT = "llm_input"
    LLM_INTEGRATION = "llm_integration"
    ANNOTATION_STORE = "annotation_store"
    SELECTED_GENES_UP = "selected_genes_up"
    SELECTED_GENES_DOWN = "selected_genes_down"
    SELECTED_UNIPROT_FIELDS = "selected_uniprot_fields"
    MAX_TOKENS = "max_tokens"
    RECENT_CHAT_WARNINGS = "recent_chat_warnings"

    ORGANISM = "organism"  # TODO: this is essentially a constant


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
