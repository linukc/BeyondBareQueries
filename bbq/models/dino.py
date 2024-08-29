import math
import types

import timm
import torch
torch.set_grad_enabled(False)
from torch import nn
from PIL import Image
from loguru import logger
from torchvision import transforms
from typing import Union, List, Tuple
import torch.nn.modules.utils as nn_utils


class DINOFeaturesExtractor:
    def __init__(self, model, load_size, stride, facet, num_patches_h, num_patches_w, **kwargs):
        self.model_type = model
        self.model = DINOFeaturesExtractor.create_model(model)
        self.model = DINOFeaturesExtractor.patch_vit_resolution(self.model, stride=stride, model_type=model)
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        if "dinov2" in model:
            self.num_reg = self.model.backbone.num_register_tokens
            self.p = self.model.backbone.patch_embed.patch_size[0]
            self.stride = self.model.backbone.patch_embed.proj.stride
        elif "dino" in model:
            self.p = self.model.patch_embed.patch_size
            self.stride = self.model.patch_embed.proj.stride
        else:
            raise NotImplementedError

        self.mean = (0.485, 0.456, 0.406) if "dino" in model else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in model else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None
        self.facet = facet
        self.load_size = load_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w

    @staticmethod
    def create_model(model_type: str):
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        if 'dinov2' in model_type:
            model = torch.hub.load('facebookresearch/dinov2', model_type)
        elif 'dino' in model_type:
            model = torch.hub.load('facebookresearch/dino:main', model_type)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                'vit_small_patch16_224': 'dino_vits16',
                'vit_small_patch8_224': 'dino_vits8',
                'vit_base_patch16_224': 'dino_vitb16',
                'vit_base_patch8_224': 'dino_vitb8'
            }
            model = torch.hub.load('facebookresearch/dino:main', model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict['head.weight']
            del temp_state_dict['head.bias']
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            try:
                pose_embed = self.pos_embed
            except:
                pose_embed = self.backbone.pos_embed
            npatch = x.shape[1] - 1
            N = pose_embed.shape[1] - 1
            if npatch == N and w == h:
                return pose_embed
            class_pos_embed = pose_embed[:, 0]
            patch_pos_embed = pose_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int, model_type: str) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        if "dinov2" in model_type:
            patch_size = list(set(model.backbone.patch_embed.patch_size))[0]
            assert type(patch_size) == int, "tested for square patch_size"
        elif "dino" in model_type:
            patch_size = model.patch_embed.patch_size
        else:
            raise NotImplementedError
        if stride == patch_size:  # nothing to do
            return model

        logger.warning(f"Interpolating position encoding for utilization of stride={stride}.")

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        if "dinov2" in model_type:
            model.backbone.patch_embed.proj.stride = stride
        elif "dino" in model_type:
            model.patch_embed.proj.stride = stride
        else:
            raise NotImplementedError
        
        if "dinov2" in model_type:
            model.backbone.interpolate_pos_encoding = types.MethodType(DINOFeaturesExtractor._fix_pos_enc(patch_size, stride), model)
        elif "dino" in model_type:
            model.interpolate_pos_encoding = types.MethodType(DINOFeaturesExtractor._fix_pos_enc(patch_size, stride), model)
        else:
            raise NotImplementedError

        return model

    def preprocess(self, image: torch.Tensor,
                   load_size: Union[int, Tuple[int, int]] = 224) -> Tuple[torch.Tensor]:
        """
        Preprocesses an image before extraction.
        :param image: torch.Tensor
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: the preprocessed image as a tensor to insert the model of shape BxCxHxW.
        """
        image = image.cpu().to(torch.uint8).numpy()
        pil_image = Image.fromarray(image)
        if "dinov2" in self.model_type:
            if load_size is not None:
                pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
                width, height = pil_image.size
                if width % self.p !=0:
                    width += self.p - width % self.p
                if height % self.p !=0:
                    height += self.p - height % self.p
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        elif "dino" in self.model_type:
            if load_size is not None:
                pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
        else:
            raise NotImplementedError
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_image = prep(pil_image)[None, ...]
        return prep_image

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        try:
            ### dino v2
            blocks = self.model.backbone.blocks
        except:
            ### dino
            blocks = self.model.blocks
        for block_idx, block in enumerate(blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def forward(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key', include_cls: bool = False) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors.
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        if facet == 'token':
            x.unsqueeze_(dim=1) # Bx1xtxd
        if "dinov2" in self.model_type and "reg" in self.model_type:
            cls = x[:, :, 0, :].unsqueeze(2)
            after_reg = x[:, :, (1+self.num_reg):, :]
            x = torch.cat([cls, after_reg], dim=2)
        
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token

        desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        #desc = desc.view(desc.shape[0], self.num_patches[0], self.num_patches[1], -1).permute(0, 3, 1, 2)
        return desc #torch.nn.functional.interpolate(desc, self.size_orig, mode="nearest")[0] # [H, W, D]
    
    def __call__(self, image):
        prep_image = self.preprocess(image, self.load_size).to("cuda" if torch.cuda.is_available() else "cpu") # Bx1xtxd
        features = self.forward(prep_image, include_cls=True, facet=self.facet) # Bx1xtx(dxh)
        return features[0, 0, 1:, :].reshape([self.num_patches_h, self.num_patches_w, -1]).cpu()

