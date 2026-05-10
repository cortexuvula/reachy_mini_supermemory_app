"""Pytest configuration: isolate tests from developer .env, add src/ to path,
and stub reachy_mini_conversation_app so tool tests can run without the full
conversation-app dependency tree (gradio, fastrtc, av, etc.).
"""

import abc
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
PROFILES_PATH = PROJECT_ROOT / "profiles"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROFILES_PATH / "supermemory") not in sys.path:
    sys.path.insert(0, str(PROFILES_PATH / "supermemory"))

os.environ["REACHY_MINI_SKIP_DOTENV"] = "1"
os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
os.environ.pop("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", None)
os.environ.pop("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", None)
os.environ.pop("SUPERMEMORY_API_KEY", None)
os.environ.pop("SUPERMEMORY_BASE_URL", None)


def _install_conversation_app_stub() -> None:
    """Provide minimal reachy_mini_conversation_app.{tools.core_tools,config} for tool tests.

    The real package brings in gradio/fastrtc/av — far too heavy for unit tests
    that only need the Tool ABC and a config object exposing
    REACHY_MINI_CUSTOM_PROFILE.
    """
    if "reachy_mini_conversation_app" in sys.modules:
        return

    pkg = types.ModuleType("reachy_mini_conversation_app")
    pkg.__path__ = []  # mark as package

    tools_pkg = types.ModuleType("reachy_mini_conversation_app.tools")
    tools_pkg.__path__ = []

    core_tools = types.ModuleType("reachy_mini_conversation_app.tools.core_tools")

    class Tool(abc.ABC):
        name: str
        description: str
        parameters_schema: dict

        @abc.abstractmethod
        async def __call__(self, deps: Any, **kwargs: Any) -> dict:
            raise NotImplementedError

    @dataclass
    class ToolDependencies:
        reachy_mini: Any = None
        movement_manager: Any = None
        camera_worker: Any = None
        vision_processor: Any = None
        head_wobbler: Any = None

    core_tools.Tool = Tool
    core_tools.ToolDependencies = ToolDependencies

    config_module = types.ModuleType("reachy_mini_conversation_app.config")

    class _Config:
        REACHY_MINI_CUSTOM_PROFILE = os.environ.get("REACHY_MINI_CUSTOM_PROFILE")

    config_module.config = _Config()

    sys.modules["reachy_mini_conversation_app"] = pkg
    sys.modules["reachy_mini_conversation_app.tools"] = tools_pkg
    sys.modules["reachy_mini_conversation_app.tools.core_tools"] = core_tools
    sys.modules["reachy_mini_conversation_app.config"] = config_module


_install_conversation_app_stub()
