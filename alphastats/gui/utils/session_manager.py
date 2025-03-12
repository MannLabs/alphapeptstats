"""Module for saving and loading session state."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytz
from cloudpickle import cloudpickle
from streamlit.runtime.state import SessionStateProxy  # noqa: TC002

from alphastats import __version__
from alphastats.gui.utils.state_keys import StateKeys
from alphastats.gui.utils.state_utils import empty_session_state, init_session_state

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
    def _copy(source: dict, target: dict | SessionStateProxy) -> None:
        """Copy a session state dictionary from `source` to `target`, only considering keys in `StateKeys`.

        The restriction to the keys in `StateKeys` is to avoid storing unnecessary data, and avoids
        potential issues when using different versions (e.g. new widgets).
        """
        target.update(
            {
                key: value
                for key, value in source.items()
                if key in StateKeys.get_values()
            }
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
        self._copy(session_state.to_dict(), state_data_to_save)
        self._save_folder_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=pytz.utc).strftime("%Y%m%d-%H%M%S")
        session_name = f"_{session_name}" if session_name else ""
        file_name = f"{_PREFIX}_{timestamp}{session_name}_v{__version__}.{_EXT}"

        file_path = self._save_folder_path / file_name
        with file_path.open("wb") as f:
            # using cloudpickle as built-in pickle does not support complex data types or lambdas
            cloudpickle.dump(state_data_to_save, f)

        return str(file_path)

    def load(self, file_name: str, session_state: SessionStateProxy) -> str:
        """Load a saved `session_state` from `file_name`, returning the file path.

        File will be looked up in `_save_folder_path`.
        """
        file_path = self._save_folder_path / file_name

        if file_path.exists():
            with file_path.open("rb") as f:
                loaded_state_data = cloudpickle.load(f)

            # clean and init first to have a defined state
            empty_session_state()
            init_session_state()
            self._copy(loaded_state_data, session_state)

            return str(file_path)

        raise ValueError(f"File {file_name} not found in {self._save_folder_path}.")
