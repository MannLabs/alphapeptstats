"""Module for saving and loading session state."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pytz
from cloudpickle import cloudpickle
from streamlit.runtime.state import SessionStateProxy  # noqa: TC002

from alphastats import __version__
from alphastats.gui.utils.state_keys import LLMKeys, StateKeys
from alphastats.gui.utils.state_utils import empty_session_state, init_session_state


class SavedSessionKeys:
    """Keys for the saved session data."""

    STATE = "state"
    META = "meta"
    VERSION = "version"
    TIMESTAMP = "timestamp"
    NAME = "name"


STATE_SAVE_FOLDER_PATH = (
    Path(__file__).absolute().parent.parent.parent.parent / "sessions"
)


# prefix and extension for pickled state
_PREFIX = "session"
_EXT = "cpkl"


class SessionManager:
    """Class for handling saving and loading session state."""

    def __init__(self, save_folder_path: str = STATE_SAVE_FOLDER_PATH):
        """Initialize the session manager with a save folder path.

        Parameters
        ----------
        save_folder_path
            absolute path to the folder where sessions are saved, defaults to `<root of repo>/sessions`

        """
        self._save_folder_path = Path(save_folder_path)

    @staticmethod
    def _safe_copy(source: dict, target: dict | SessionStateProxy) -> None:
        """Copy a session state dictionary from `source` to `target`, only considering keys in `StateKeys`.

        The restriction to the keys in `StateKeys` is to avoid storing unnecessary data, and avoids
        potential issues when using different versions (e.g. new widgets).

        Also, the LLM client is removed from the session state to avoid pickling issues.
        """
        keys_to_save = StateKeys.get_values()
        keys_to_save.remove(StateKeys.OPENAI_API_KEY)  # do not store key on disk

        source = source.copy()

        for chat in source.get(StateKeys.LLM_CHATS, {}).values():
            if (llm_integration := chat.get(LLMKeys.LLM_INTEGRATION)) is not None:
                # cannot pickle LLM client due to SSLContext object
                llm_integration.client_wrapper = None

        target.update(
            {key: value for key, value in source.items() if key in keys_to_save}
        )

    @staticmethod
    def get_saved_sessions(save_folder_path: str) -> list[str]:
        """Get a list of saved session file names from the `save_folder_path`."""
        return sorted(
            [f.name for f in Path(save_folder_path).glob(f"*.{_EXT}") if f.is_file()],
            reverse=True,
        )

    def save(self, session_state: SessionStateProxy, session_name: str = "") -> str:
        """Save the current `session_state` to a file, returning the file path.

        Only considering keys in `StateKeys` are saved.
        """
        state_data_to_save = {}
        self._safe_copy(session_state.to_dict(), state_data_to_save)
        self._save_folder_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=pytz.utc).strftime("%Y%m%d-%H%M%S")
        session_name = f"_{session_name}" if session_name else ""
        file_name = f"{_PREFIX}_{timestamp}{session_name}.{_EXT}"

        data_to_dump = {
            SavedSessionKeys.STATE: state_data_to_save,
            SavedSessionKeys.META: {
                SavedSessionKeys.VERSION: __version__,
                SavedSessionKeys.TIMESTAMP: timestamp,
                SavedSessionKeys.NAME: session_name,
            },
        }

        file_path = self._save_folder_path / file_name
        with file_path.open("wb") as f:
            # using cloudpickle as built-in pickle does not support complex data types or lambdas
            cloudpickle.dump(data_to_dump, f)

        return str(file_path)

    def load(self, file_name: str, session_state: SessionStateProxy) -> str:
        """Load a saved `session_state` from `file_name`, returning the file path.

        File will be looked up in `_save_folder_path`.
        """
        file_path = self._save_folder_path / file_name

        if file_path.exists():
            with file_path.open("rb") as f:
                loaded_data = cloudpickle.load(f)

            loaded_state_data = loaded_data[SavedSessionKeys.STATE]

            logging.info(
                f"Loaded session {file_name} from {self._save_folder_path}: {loaded_data[SavedSessionKeys.META]}"
            )

            if (
                version := loaded_data[SavedSessionKeys.META][SavedSessionKeys.VERSION]
            ) != __version__:
                logging.warning(
                    f"Version mismatch: Session {file_name} was saved with version {version}, but current version is {__version__}."
                    f"This might lead to unexpected behavior."
                )

            # clean and init first to have a defined state
            empty_session_state()
            init_session_state()
            self._safe_copy(loaded_state_data, session_state)

            return str(file_path)

        raise ValueError(f"File {file_name} not found in {self._save_folder_path}.")
