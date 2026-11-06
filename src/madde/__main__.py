import sys

from madde import db
from madde.config import get_settings


def main() -> None:
    if sys.argv[1:] == ["migrate"]:
        for name in db.migrate(get_settings().database_url):
            print(f"applied {name}")
        return
    print("usage: python -m madde migrate", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
