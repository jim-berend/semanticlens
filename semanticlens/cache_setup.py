import logging

from joblib import Memory

logger = logging.getLogger(__name__)


class CacheProxy:
    """A proxy that delegates caching to a real Memory object.

    This allows the real cache to be configured at runtime, after modules
    have already been imported and decorators have been evaluated.
    """

    def __init__(self):
        # Start with a dummy cache that does nothing.
        # TODO check if env is set!
        self._instance = Memory(location=None)
        self._is_configured = False

    @property
    def cache_dir(self):
        """Get the cache directory."""
        return self._instance.location

    def configure(self, location, **kwargs):
        """Creates the real cache instance."""
        if self._is_configured:
            logger.warning("Caching is already configured (%s). Ignoring subsequent call.", self.cache_dir)
            return

        self._instance = Memory(location=location, **kwargs)
        self._is_configured = True
        logger.info(f"SemanticLens: Caching enabled. Storing results in '{location}'.")

    def cache(self, *args, **kwargs):
        """The decorator will call this method."""
        return self._instance.cache(*args, **kwargs)


memory = CacheProxy()
