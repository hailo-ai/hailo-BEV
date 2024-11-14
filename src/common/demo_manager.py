class DemoManager:
    """
    A class to manage the termination state in a multi-process or multi-threaded environment.

    This class is designed to manage a shared termination flag across processes or threads,
    enabling the program to gracefully terminate when the flag is set. It utilizes a
    multiprocessing `Value` to ensure that the termination signal can be shared safely.

    Args:
        manager (multiprocessing.Manager): A manager instance to handle
        shared state between processes.

    Attributes:
        terminate (multiprocessing.Value): A shared flag used to
        indicate whether the program should terminate.

    Methods:
        set_terminate(): Sets the termination flag to True.
        get_terminate(): Returns the current value of the termination flag (True or False).
    """
    def __init__(self, manager):
        """
        Initializes the demo_manager instance with a shared termination flag.

        Args:
            manager (multiprocessing.Manager): A manager instance to manage the shared flag.
        """
        self.terminate = manager.Value('b', False)

    def set_terminate(self) -> None:
        """
        Sets the termination flag to True, signaling that the program should terminate.

        This method updates the shared termination flag to indicate that the program
        should stop its operations and begin shutting down.
        """
        self.terminate.value = True

    def get_terminate(self) -> bool:
        """
        Retrieves the current value of the termination flag.

        Returns:
            bool: The current value of the termination flag.
                  - True if the program should terminate.
                  - False if the program should continue running.
        """
        return self.terminate.value
    