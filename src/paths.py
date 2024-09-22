import pathlib
import uuid

# Define constant paths
CODE_ROOT: pathlib.Path = pathlib.Path(__file__).parent
REPO_ROOT: pathlib.Path = CODE_ROOT.parent
DATA_ROOT: pathlib.Path = REPO_ROOT / "data"
LOG_DIR: pathlib.Path = REPO_ROOT / "logs"

# Create log directory if it doesn't exist
LOG_DIR.mkdir(exist_ok=True)


def random_checkpoint_dir() -> str:
    """
    Generate a random directory path for checkpoints.

    This function creates a unique directory path within the 'checkpoints' folder
    of the repository root. The directory name is a randomly generated UUID.

    Returns:
        str: A string representation of the path to a new, unique checkpoint directory.

    Raises:
        AssertionError: If the generated directory already exists (which is highly unlikely).
    """
    rand_dir: pathlib.Path = REPO_ROOT / "checkpoints" / str(uuid.uuid4())
    assert not rand_dir.exists(), f"Random directory {rand_dir} already exists"
    return str(rand_dir)
