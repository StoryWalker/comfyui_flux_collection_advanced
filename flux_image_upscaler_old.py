import folder_paths

class FluxImageUpscaler:

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod

        def INPUT_TYPES(s):

            return {"required": {
                                "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                                "image": ("IMAGE",),
                                "upscale_method": (s.upscale_methods,),
                                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),

                                }
                    }

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "load_model"
        CATEGORY = "fluxCollection"


        def load_model(self, model_name, image, upscale_method, scale_by):

            model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
            sd = comfy.utils.load_torch_file(model_path, safe_load=True)
            if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
                sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
                
            out = ModelLoader().load_from_state_dict(sd).eval()



            if not isinstance(out, ImageModelDescriptor):

            raise Exception("Upscale model must be a single-image model.")



        #return (out, )



        upscale_model = out



        #def upscale(self, upscale_model, image):

        device = model_management.get_torch_device()



        memory_required = model_management.module_size(upscale_model.model)

        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate

        memory_required += image.nelement() * image.element_size()

        model_management.free_memory(memory_required, device)



        upscale_model.to(device)

        in_img = image.movedim(-1,-3).to(device)



        tile = 512

        overlap = 32



        oom = True

        while oom:

        try:

        steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)

        pbar = comfy.utils.ProgressBar(steps)

        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)

        oom = False

        except model_management.OOM_EXCEPTION as e:

        tile //= 2

        if tile < 128:

        raise e



        upscale_model.to("cpu")

        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)

        #return (s,)



        #def upscale(self, image, upscale_method, scale_by):

        image = s

        samples = image.movedim(-1,1)

        width = round(samples.shape[3] * scale_by)

        height = round(samples.shape[2] * scale_by)

        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")

        s = s.movedim(1,-1)

        return (s,)