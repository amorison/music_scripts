from . import config


def main() -> None:
    """Implement mutools entry point."""
    config.parse_args_and_run()


if __name__ == "__main__":
    main()
