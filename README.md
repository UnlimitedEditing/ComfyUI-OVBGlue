# ComfyUI-OVBGlue

Two small helper nodes for Open Vision Banana (FunAILab, FLUX.2 klein base 9B fine-tune) in ComfyUI.
No weights, no extra dependencies.

- `OVBPromptBuilder`: mode + plain text -> the exact task prompt the model was trained on (templates from the
  FunAILab model card), the segmentation colour map, and whether the source image is used.
- `OVBRestoreSize`: resizes perception outputs back to the source image size (NEAREST for masks, BILINEAR for
  depth and normals), as the model card does after sampling.

Modes: auto, edit, t2i, semseg, refseg, instseg, depth, normals, raw.
