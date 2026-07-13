"""CLI entry: ``poetry run python -m viz``."""

from viz.app import create_app

HOST = "127.0.0.1"
PORT = 8765


def main() -> None:
    """Launch local trajectory viewer."""
    app = create_app()
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
