"""Module for saving and loading session state."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytz
from cloudpickle import cloudpickle
from streamlit.runtime.state import SessionStateProxy  # noqa: TC002

from alphastats.gui.utils.state_keys import StateKeys

STATE_SAVE_FOLDER = Path(__file__).absolute().parent.parent.parent.parent / "sessions"


# prefix and extension for pickled state
_PREFIX = "session_"
_EXT = "cpkl"


class SessionManager:
    """Class for handling saving and loading session state."""

    def __init__(self, save_path: str = STATE_SAVE_FOLDER):
        """Initialize the session manager with a save folder path."""
        self._save_folder_path = Path(save_path)

        self._save_folder_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _copy(source: dict, target: dict | SessionStateProxy) -> None:
        """Copy a session state from source to target, only considering custom keys."""
        target.update(
            {
                key: value
                for key, value in source.items()
                if key in StateKeys.get_values()
            }
        )

    @staticmethod
    def get_saved_sessions(save_folder_path: str) -> list[str]:
        """Get a list of saved session files in the `save_folder_path`."""
        try:
            return sorted(
                [
                    f.name
                    for f in Path(save_folder_path).glob(f"*.{_EXT}")
                    if f.is_file()
                ],
                reverse=True,
            )
        except FileNotFoundError as e:
            raise ValueError(f"The folder {save_folder_path} does not exist.") from e

    def save(self, session_state: SessionStateProxy) -> str:
        """Save the current session state to a file."""
        target = {}
        self._copy(session_state.to_dict(), target)

        timestamp = datetime.now(tz=pytz.utc).strftime("%Y%m%d-%H%M%S")
        file_name = f"{_PREFIX}{timestamp}.{_EXT}"

        file_path = self._save_folder_path / file_name
        with file_path.open("wb") as f:
            # built-in pickle does not support the complext data types or lambdas
            cloudpickle.dump(target, f)

        return str(file_path)

    def load(self, file_name: str, session_state: SessionStateProxy) -> str:
        """Load a saved session state from `file_name`."""
        file_path = self._save_folder_path / file_name

        if file_path.exists():
            with file_path.open("rb") as f:
                loaded_state = cloudpickle.load(f)
                self._copy(loaded_state, session_state)
        else:
            raise ValueError(f"File {file_name} not found in {self._save_folder_path}.")

        return str(file_path)
