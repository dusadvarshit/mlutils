import logging

# from pydantic import Field
# from pydantic.dataclasses import dataclass


class CustomLogger:
    def __init__(self, name, level="INFO") -> None:
        """Initialize the logger after class instantiation"""
        self.name = name
        self.level = level

        self.logger = logging.getLogger(self.name)
        self._set_level(self.level)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _set_level(self, level: str) -> None:
        """Set the logging level

        Args:
            level (str): Desired logging level
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        self.logger.setLevel(level_map.get(level.upper(), logging.INFO))

    def change_level(self, new_level: str) -> None:
        """Change the logging level

        Args:
            new_level (str): New logging level to set
        """
        self.level = new_level
        self._set_level(new_level)

    def get_logger(self) -> logging.Logger:
        """Get the logger instance

        Returns:
            logging.Logger: Configured logger instance
        """
        return self.logger
