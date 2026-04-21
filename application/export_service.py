# -*- coding: utf-8 -*-
import logging
try:
    from domain.models import VideoExportManifest
    from infrastructure.video_io_adapter import ImageIOSaver
except ImportError:
    from ..domain.models import VideoExportManifest
    from ..infrastructure.video_io_adapter import ImageIOSaver

logger = logging.getLogger(__name__)

class ExportVideoService:
    """
    Application Service:
    Coordinates folder organization and video writing.
    """
    
    def __init__(self, saver=None):
        self.saver = saver if saver is not None else ImageIOSaver()

    def export_video(self, manifest: VideoExportManifest, frames: any) -> str:
        """
        Orchestrates the export process by resolving paths through infrastructure
        and executing the save operation.
        """
        # 1. Resolve path through Infra Adapter
        full_path = self.saver.resolve_full_path(
            manifest.filename_prefix, 
            manifest.index, 
            manifest.custom_path
        )

        # 2. Execute Save
        self.saver.save_mp4(full_path, frames, manifest.fps)
        
        return full_path
