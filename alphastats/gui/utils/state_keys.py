"""Keys functions for the session state."""

from alphastats.dataset.keys import ConstantsClass


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
