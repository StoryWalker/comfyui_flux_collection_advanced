# -*- coding: utf-8 -*-
"""
Tests for [LTX] Legacy Nodes (Thin Hexagonal Wrappers).

Estos nodos mantienen la interfaz legacy pero delegan internamente a la
arquitectura hexagonal (LTXPipelineAdapter, LTXVideoGenerationService, ExportVideoService).
Los mocks deben apuntar a esos componentes, no al backend directo.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, ANY
import torch

# CRITICAL WORKAROUND: Force a valid ModuleSpec for imageio to prevent ValueError: imageio.__spec__ is None
try:
    import imageio
    import importlib.machinery
    if "imageio" in sys.modules:
        mod = sys.modules["imageio"]
        if getattr(mod, "__spec__", None) is None:
            mod.__spec__ = importlib.machinery.ModuleSpec(
                name="imageio",
                loader=None,
                origin=getattr(mod, "__file__", "built-in")
            )
except Exception:
    pass

sys.path.append(os.getcwd())

import folder_paths
if not hasattr(folder_paths, "get_output_directory"):
    folder_paths.get_output_directory = lambda: "/tmp"
if not hasattr(folder_paths, "get_filename_list"):
    folder_paths.get_filename_list = lambda x: []
if not hasattr(folder_paths, "get_full_path"):
    folder_paths.get_full_path = lambda x, y: f"/tmp/{y}"

from ltx_nodes import LTXDistilledGGUFPipelineLoader, LTXFastPipelineLoader, LTXVideoSampler


class TestLTXNodes(unittest.TestCase):

    @patch("ltx_nodes.resolve_ltx_path")
    @patch("ltx_nodes.LTXPipelineAdapter")
    @patch("os.path.exists", return_value=True)
    def test_gguf_loader_initializes_pipeline(self, mock_exists, mock_adapter_cls, mock_resolve):
        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter
        mock_pipeline = MagicMock()
        mock_adapter.load_pipeline.return_value = mock_pipeline
        mock_resolve.side_effect = lambda filename, folder_type=None: f"resolved/{filename}"

        loader = LTXDistilledGGUFPipelineLoader()
        result = loader.load_pipeline(
            gguf_checkpoint="dummy_gguf.gguf",
            gemma_directory="dummy_gemma",
            spatial_upscaler="dummy_upscaler.safetensors",
            vae_video="dummy_vae_v.safetensors",
            audio_vae="dummy_vae_a.safetensors",
            connector="dummy_connector.safetensors",
            device="cpu"
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_pipeline)
        mock_adapter.load_pipeline.assert_called_once()
        config = mock_adapter.load_pipeline.call_args[0][0]
        self.assertEqual(config.checkpoint_path, "resolved/dummy_gguf.gguf")
        self.assertEqual(config.gemma_root, "resolved/dummy_gemma")
        self.assertEqual(config.upsampler_path, "resolved/dummy_upscaler.safetensors")
        self.assertEqual(config.vae_video_path, "resolved/dummy_vae_v.safetensors")
        self.assertEqual(config.audio_vae_path, "resolved/dummy_vae_a.safetensors")
        self.assertEqual(config.connector_path, "resolved/dummy_connector.safetensors")
        self.assertEqual(config.pipeline_type, "distilled_gguf")
        self.assertEqual(config.device, "cpu")

    @patch("ltx_nodes.resolve_ltx_path")
    @patch("ltx_nodes.LTXPipelineAdapter")
    @patch("os.path.exists", return_value=True)
    def test_fast_loader_initializes_pipeline(self, mock_exists, mock_adapter_cls, mock_resolve):
        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter
        mock_pipeline = MagicMock()
        mock_adapter.load_pipeline.return_value = mock_pipeline
        mock_resolve.side_effect = lambda filename, folder_type=None: f"resolved/{filename}"

        loader = LTXFastPipelineLoader()
        result = loader.load_pipeline(
            checkpoint="dummy_fast.safetensors",
            gemma_directory="dummy_gemma",
            spatial_upscaler="dummy_upscaler.safetensors",
            device="cpu"
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_pipeline)
        mock_adapter.load_pipeline.assert_called_once()
        config = mock_adapter.load_pipeline.call_args[0][0]
        self.assertEqual(config.checkpoint_path, "resolved/dummy_fast.safetensors")
        self.assertEqual(config.gemma_root, "resolved/dummy_gemma")
        self.assertEqual(config.upsampler_path, "resolved/dummy_upscaler.safetensors")
        self.assertEqual(config.pipeline_type, "fast")
        self.assertEqual(config.device, "cpu")

    @patch("ltx_nodes.LTXVideoGenerationService")
    @patch("ltx_nodes.LTXPipelineAdapter")
    def test_sampler_without_image_converts_output_range(self, mock_adapter_cls, mock_service_cls):
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service
        fake_video = torch.rand(4, 64, 64, 3, dtype=torch.float32)
        mock_service.generate.return_value = (fake_video, None)

        sampler = LTXVideoSampler()
        result = sampler.sample(
            ltx_pipeline=MagicMock(),
            prompt="a cat walking",
            width=256,
            height=256,
            num_frames=9,
            frame_rate=8.0,
            seed=42,
            image=None,
            strength=1.0
        )

        self.assertEqual(len(result), 2)
        output_tensor, audio_data = result
        self.assertEqual(output_tensor.shape, torch.Size([4, 64, 64, 3]))
        self.assertEqual(output_tensor.dtype, torch.float32)
        self.assertTrue(output_tensor.max() <= 1.0)
        self.assertTrue(output_tensor.min() >= 0.0)
        self.assertIsNone(audio_data)

        # Verify service was called with correct settings
        mock_service.generate.assert_called_once()
        pipeline_arg, settings_arg = mock_service.generate.call_args[0]
        self.assertEqual(settings_arg.prompt, "a cat walking")
        self.assertEqual(settings_arg.width, 256)
        self.assertEqual(settings_arg.height, 256)
        self.assertEqual(settings_arg.num_frames, 9)
        self.assertEqual(settings_arg.frame_rate, 8.0)
        self.assertEqual(settings_arg.seed, 42)
        self.assertEqual(settings_arg.strength, 1.0)
        self.assertEqual(settings_arg.image_path, "")

    @patch("ltx_nodes.LTXVideoGenerationService")
    @patch("ltx_nodes.LTXPipelineAdapter")
    def test_sampler_with_image_creates_temp_file(self, mock_adapter_cls, mock_service_cls):
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service
        fake_video = torch.rand(2, 64, 64, 3, dtype=torch.float32)
        mock_service.generate.return_value = (fake_video, None)

        fake_image = torch.rand(1, 256, 256, 3, dtype=torch.float32)

        sampler = LTXVideoSampler()
        with patch("os.unlink") as mock_unlink:
            result = sampler.sample(
                ltx_pipeline=MagicMock(),
                prompt="a bird flying",
                width=256,
                height=256,
                num_frames=9,
                frame_rate=8.0,
                seed=42,
                image=fake_image,
                strength=0.8
            )

            # Verify temporary file unlink was called for the conditional image path
            mock_unlink.assert_any_call(ANY)

            # Verify service was called with image conditioning
            mock_service.generate.assert_called_once()
            _pipeline_arg, settings_arg = mock_service.generate.call_args[0]
            self.assertTrue(settings_arg.use_image_conditioning)
            self.assertEqual(settings_arg.strength, 0.8)

    @patch("folder_paths.get_output_directory")
    @patch("ltx_nodes.ExportVideoService")
    def test_saver_saves_video_and_returns_ui(self, mock_service_cls, mock_get_output):
        mock_get_output.return_value = "dummy_output_dir"
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service
        mock_service.export_video.return_value = os.path.join("dummy_output_dir", "2024-01-01", "videos", "test_ltx_0000_120000.mp4")

        fake_images = torch.rand(2, 64, 64, 3, dtype=torch.float32)

        from ltx_nodes import LTXVideoSaver
        saver = LTXVideoSaver()
        result = saver.save_video(
            images=fake_images,
            fps=24,
            filename_prefix="test_ltx"
        )

        mock_service.export_video.assert_called_once()
        manifest, frames = mock_service.export_video.call_args[0]
        self.assertEqual(manifest.filename_prefix, "test_ltx")
        self.assertEqual(manifest.fps, 24)

        self.assertIn("ui", result)
        self.assertIn("gifs", result["ui"])
        self.assertEqual(len(result["ui"]["gifs"]), 1)
        self.assertEqual(result["ui"]["gifs"][0]["type"], "output")


if __name__ == '__main__':
    unittest.main()
