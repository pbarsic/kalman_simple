import logging
from pathlib import Path
import time


def logger_setup(
    filename: str = None,
    log_folder: Path = None,
    console_output=True,
    overwrite_old=False,
    debug_to_console=False,
) -> Path:
    """
    utility for creating a logger
    """

    if log_folder is None:
        log_folder = Path("/tmp")

    if filename is None:
        logger_filename = log_folder / f"log_{time.strftime('%Y%m%d_%H.%M.%s')}.txt"
    else:
        logger_filename = log_folder / filename

    if not log_folder.is_dir():
        log_folder.mkdir(parents=True)

    if overwrite_old and logger_filename.exists():
        logger_filename.unlink()

    # basicConfig has to have the lowest log level
    logging.basicConfig(
        filename=str(logger_filename),
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if console_output or debug_to_console:
        console_handler = logging.StreamHandler()
        if debug_to_console:  # TODO figure out why this doesn't work
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
        c_format = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(c_format)
        logging.getLogger().addHandler(console_handler)

    logging.info(f"logging filename: {logger_filename}")
    return logger_filename


if __name__ == "__main__":
    """
    example of how to use the logger
    """

    import argparse

    # import imread_alpha

    parser = argparse.ArgumentParser()
    program_name = str(Path(parser.prog).stem)
    parser.add_argument(
        "-l", "--log", default="/tmp", type=str, help="destination for logfile"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="display debug to console"
    )
    parser.add_argument(
        "-c", "--clobber", action="store_true", help="overwrite old output"
    )
    args = parser.parse_args()
    logger_setup(
        filename=program_name,
        log_folder=Path(args.log),
        debug_to_console=args.debug,
        overwrite_old=args.clobber,
    )
    logging.warning("This is a warning")
    logging.error("This is an error")
    logging.debug("This is a debug")
