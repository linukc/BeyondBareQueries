import os
from typing import Any, Generator, List

import torch
torch.set_grad_enabled(False)
import numpy as np
from mobilesamv2 import sam_model_registry, SamPredictor
from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel


class MobileSAMGenerator:
    def __init__(self, weights_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_path = weights_path
        
        prompt_guided_path = os.path.join(self.weights_path,
                                "Prompt_guided_Mask_Decoder.pt")
        obj_model_path = os.path.join(self.weights_path,
                                "ObjectAwareModel.pt")
        self.ObjAwareModel = ObjectAwareModel(obj_model_path)
        
        PromptGuidedDecoder = sam_model_registry['PromptGuidedDecoder'](prompt_guided_path)
        self.mobilesamv2 = sam_model_registry['vit_h']()
        self.mobilesamv2.prompt_encoder = PromptGuidedDecoder['PromtEncoder']
        self.mobilesamv2.mask_decoder = PromptGuidedDecoder['MaskDecoder']
        image_encoder = sam_model_registry["efficientvit_l2"](
            os.path.join(self.weights_path, 'l2.pt'))
        self.mobilesamv2.image_encoder = image_encoder
        self.mobilesamv2.to(device=self.device)
        self.mobilesamv2.eval()
        self.predictor = SamPredictor(self.mobilesamv2)

    def __call__(self, image):
        """
        image: [H, W, 3] numpy array
        """
        obj_results = self.ObjAwareModel(image, 
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9)
        self.predictor.set_image(image)
        input_boxes1 = obj_results[0].boxes.xyxy
        input_boxes = input_boxes1.cpu().numpy()
        input_boxes = self.predictor.transform.apply_boxes(input_boxes,
                self.predictor.original_size)
        h_scale = self.predictor.original_size[0] / self.predictor.input_size[0]
        w_scale = self.predictor.original_size[1] / self.predictor.input_size[1]
        ref = np.array([h_scale, w_scale, h_scale, w_scale])
        output_boxes = torch.from_numpy(input_boxes * ref)
        input_boxes = torch.from_numpy(input_boxes).cuda()
        image_embedding = self.predictor.features
        image_embedding = torch.repeat_interleave(image_embedding, 320, dim=0)
        prompt_embedding = self.mobilesamv2.prompt_encoder.get_dense_pe()
        prompt_embedding = torch.repeat_interleave(prompt_embedding, 320, dim=0)

        sam_mask = []
        for (boxes,) in self.batch_iterator(320, input_boxes):
            with torch.no_grad():
                image_embedding=image_embedding[0:boxes.shape[0],:,:,:]
                prompt_embedding=prompt_embedding[0:boxes.shape[0],:,:,:]
                sparse_embeddings, dense_embeddings = self.mobilesamv2.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,)
                low_res_masks, iou_pred = self.mobilesamv2.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=prompt_embedding,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )
                low_res_masks = self.predictor.model.postprocess_masks(low_res_masks,
                                    self.predictor.input_size, self.predictor.original_size)
                sam_mask_pre = (low_res_masks > self.mobilesamv2.mask_threshold)*1.0
                sam_mask.append(sam_mask_pre.squeeze(1))
        sam_mask = torch.cat(sam_mask)
        return output_boxes, sam_mask, iou_pred

    def batch_iterator(self, batch_size: int, *args) -> Generator[List[Any], None, None]:
        assert len(args) > 0 and all(
            len(a) == len(args[0]) for a in args
        ), "Batched iteration must have inputs of all the same size."
        n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
        for b in range(n_batches):
            yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]
