# -*- coding: utf-8 -*-
import os
import traceback
import datetime

class ErrorLoggingAdapter:
    """
    Infrastructure Adapter to capture and persist execution errors 
    for Gemini CLI autonomous diagnosis.
    """
    def __init__(self):
        # We use the path you prefer for error monitoring
        self.error_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "references", "error.txt")
        os.makedirs(os.path.dirname(self.error_file), exist_ok=True)

    def log_error(self, node_name: str, exception: Exception, context: dict = None):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_msg = f"[{timestamp}] ERROR in {node_name}\n"
        error_msg += f"Exception: {str(exception)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}\n"
        if context:
            error_msg += f"Context: {context}\n"
        error_msg += "-"*50 + "\n"

        with open(self.error_file, "a", encoding="utf-8") as f:
            f.write(error_msg)

def hex_error_handler(func):
    """Decorator to wrap node execution and capture errors for Gemini."""
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            adapter = ErrorLoggingAdapter()
            adapter.log_error(self.__class__.__name__, e, kwargs)
            # Re-raise to let ComfyUI show the red box, but we already captured it
            raise e
    return wrapper
