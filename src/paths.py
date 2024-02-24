import pathlib
import uuid

CODE_ROOT = pathlib.Path(__file__).parent
REPO_ROOT = CODE_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"

LOG_DIR = REPO_ROOT / "logs"
if not LOG_DIR.exists():  # in case LOG_DIR is a symlink
    LOG_DIR.mkdir(exist_ok=True, parents=True)


def random_checkpoint_dir():
    rand_dir = REPO_ROOT / "checkpoints" / str(uuid.uuid4())
    assert not rand_dir.exists()
    return str(rand_dir)
