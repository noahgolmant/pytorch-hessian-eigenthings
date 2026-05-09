import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    try:
        import curvlinops  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="curvlinops not installed")
        for item in items:
            if "curvlinops" in item.keywords:
                item.add_marker(skip)

    try:
        import transformers  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="transformers not installed")
        for item in items:
            if "transformers" in item.keywords:
                item.add_marker(skip)

    try:
        import transformer_lens  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="transformer_lens not installed")
        for item in items:
            if "transformer_lens" in item.keywords:
                item.add_marker(skip)
