from typing import Iterator

from tqdm import tqdm


def progress_range(*args, desc: str = "Processing", **kwargs) -> Iterator[int]:
    """
    A range function with progress bar using tqdm.

    Args:
        *args: Arguments passed to range() (start, stop, step)
        desc: Description for the progress bar
        **kwargs: Additional keyword arguments passed to tqdm

    Returns:
        Iterator yielding integers with progress display

    Examples:
        # Simple range with progress
        for i in progress_range(100):
            # do something
            pass

        # Range with start, stop, step
        for i in progress_range(0, 100, 2, desc="Even numbers"):
            # do something
            pass
    """
    range_obj = range(*args)
    return tqdm(range_obj, desc=desc, **kwargs)
