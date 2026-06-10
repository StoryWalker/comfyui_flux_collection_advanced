# -*- coding: utf-8 -*-
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
    @patch("ltx_nodes.LTXDistilledGGUFVideoPipeline")
    @patch("os.path.exists", return_value=True)
    def test_gguf_loader_initializes_pipeline(self, mock_exists, mock_gguf_pipeline, mock_resolve):
        mock_instance = MagicMock()
        mock_gguf_pipeline.create.return_value = mock_instance
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
        self.assertEqual(result[0], mock_instance)
        mock_gguf_pipeline.create.assert_called_once_with(
            checkpoint_path="resolved/dummy_gguf.gguf",
            gemma_root="resolved/dummy_gemma",
            upsampler_path="resolved/dummy_upscaler.safetensors",
            device=torch.device("cpu"),
            vae_video_path="resolved/dummy_vae_v.safetensors",
            audio_vae_path="resolved/dummy_vae_a.safetensors",
            connector_path="resolved/dummy_connector.safetensors"
        )

    @patch("ltx_nodes.resolve_ltx_path")
    @patch("ltx_nodes.LTXFastVideoPipeline")
    @patch("os.path.exists", return_value=True)
    def test_fast_loader_initializes_pipeline(self, mock_exists, mock_fast_pipeline, mock_resolve):
        mock_instance = MagicMock()
        mock_fast_pipeline.create.return_value = mock_instance
        mock_resolve.side_effect = lambda filename, folder_type=None: f"resolved/{filename}"

        loader = LTXFastPipelineLoader()
        result = loader.load_pipeline(
            checkpoint="dummy_fast.safetensors",
            gemma_directory="dummy_gemma",
            spatial_upscaler="dummy_upscaler.safetensors",
            device="cpu"
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_instance)
        mock_fast_pipeline.create.assert_called_once_with(
            checkpoint_path="resolved/dummy_fast.safetensors",
            gemma_root="resolved/dummy_gemma",
            upsampler_path="resolved/dummy_upscaler.safetensors",
            device=torch.device("cpu")
        )

    @patch("ltx_nodes.default_tiling_config")
    def test_sampler_without_image_converts_output_range(self, mock_tiling):
        mock_pipeline = MagicMock()
        # Mocking VAE output video: a 4-frame video of shape [4, 64, 64, 3] with values in [0, 255]
        fake_video = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)
        mock_pipeline._run_inference.return_value = (fake_video, None)

        sampler = LTXVideoSampler()
        result = sampler.sample(
            ltx_pipeline=mock_pipeline,
            prompt="a cat walking",
            width=256,
            height=256,
            num_frames=9,
            frame_rate=8.0,
            seed=42,
            image=None,
            strength=1.0
        )

        self.assertEqual(len(result), 1)
        output_tensor = result[0]
        # Verify shape remains [F, H, W, C]
        self.assertEqual(output_tensor.shape, torch.Size([4, 64, 64, 3]))
        # Verify it has been scaled down to [0.0, 1.0] float32
        self.assertEqual(output_tensor.dtype, torch.float32)
        self.assertTrue(output_tensor.max() <= 1.0)
        self.assertTrue(output_tensor.min() >= 0.0)

    @patch("ltx_nodes.default_tiling_config")
    def test_sampler_with_image_creates_temp_file(self, mock_tiling):
        mock_pipeline = MagicMock()
        fake_video = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
        mock_pipeline._run_inference.return_value = (fake_video, None)

        # Mock image: a single frame of shape [1, 256, 256, 3] in float32 [0.0, 1.0]
        fake_image = torch.rand(1, 256, 256, 3, dtype=torch.float32)

        sampler = LTXVideoSampler()
        
        # We check that a temporary file path was created and deleted
        with patch("os.unlink") as mock_unlink:
            result = sampler.sample(
                ltx_pipeline=mock_pipeline,
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
            
            # Verify images list was supplied with strength
            called_args = mock_pipeline._run_inference.call_args[1]["images"]
            self.assertEqual(len(called_args), 1)
            self.assertEqual(called_args[0]["strength"], 0.8)

    @patch("folder_paths.get_output_directory")
    @patch("imageio.mimwrite")
    def test_saver_saves_video_and_returns_ui(self, mock_mimwrite, mock_get_output):
        mock_get_output.return_value = "dummy_output_dir"
        
        # Simula entrada IMAGE de ComfyUI (2 frames, 64x64, RGB)
        fake_images = torch.rand(2, 64, 64, 3, dtype=torch.float32)
        
        from ltx_nodes import LTXVideoSaver
        saver = LTXVideoSaver()
        result = saver.save_video(
            images=fake_images,
            fps=24,
            filename_prefix="test_ltx"
        )
        
        # Comprobar llamada mimwrite
        mock_mimwrite.assert_called_once()
        called_args = mock_mimwrite.call_args[0]
        self.assertEqual(len(called_args[1]), 2) # 2 frames
        self.assertEqual(called_args[1][0].shape, (64, 64, 3))
        
        # Comprobar retorno de metadatos UI
        self.assertIn("ui", result)
        self.assertIn("gifs", result["ui"])
        self.assertEqual(len(result["ui"]["gifs"]), 1)
        self.assertEqual(result["ui"]["gifs"][0]["type"], "output")


if __name__ == '__main__':
    unittest.main()
