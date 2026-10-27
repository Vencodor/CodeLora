"""Fictional fixture: never contacts a server."""

BATCH_LIMIT = 8


def count_batch(paths):
    """Return the count, rejecting batches larger than BATCH_LIMIT."""
    if len(paths) > BATCH_LIMIT:
        raise ValueError("batch limit exceeded")
    return len(paths)
