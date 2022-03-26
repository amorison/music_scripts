from . import config


def main() -> None:
    """Implement mutools entry point."""
    config.parse_args()


if __name__ == "__main__":
    main()
