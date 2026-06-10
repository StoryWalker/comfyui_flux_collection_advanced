# -*- coding: utf-8 -*-
import os
import traceback
import datetime
import time
import logging

logger = logging.getLogger("FluxCollectionAdvanced")

class ErrorLoggingAdapter:
    """
    Infrastructure Adapter to capture and persist execution errors 
    and log performance metrics for Gemini CLI autonomous diagnosis.
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
    """Decorator to wrap node execution and capture errors for Gemini diagnostic log."""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        node_name = self.__class__.__name__
        try:
            result = func(self, *args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"\033[96m[HEX Performance] {node_name} ejecutado con éxito en {elapsed_time:.2f}s\033[0m")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"\033[91m[HEX Error] {node_name} falló tras {elapsed_time:.2f}s: {e}\033[0m")
            adapter = ErrorLoggingAdapter()
            adapter.log_error(node_name, e, kwargs)
            raise e
    return wrapper


# --- REGISTRO DEL ENDPOINT PARA LOGS UNIFICADOS DEL PIPELINE ---
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/hex/log_pipeline")
    async def hex_log_pipeline(request):
        try:
            data = await request.json()
            log_content = data.get("log", "")
            
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            timestamp_file = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
            log_filename = f"{timestamp_file}_execution.log"
            log_path = os.path.join(logs_dir, log_filename)
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_content + "\n" + "="*60 + "\n\n")
                
            return web.json_response({"status": "success"})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)
except Exception as e:
    logger.warning(f"[HEX] No se pudo registrar la ruta de logs de pipeline: {e}")
