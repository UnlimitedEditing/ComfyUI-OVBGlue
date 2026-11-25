"""Glue nodes for Open Vision Banana (FunAILab, FLUX.2 klein base 9B fine-tune).

OVBPromptBuilder: turns (mode, plain user text) into the exact task prompt the model was trained on
                  (templates copied from the FunAILab model card) and decides whether the source image is used.
OVBRestoreSize:   resizes perception outputs back to the source image size (NEAREST for masks, BILINEAR for
                  depth / normals), as the model card does after sampling.
"""
import re

# mode -> canonical name. Anything else raises with the list of valid modes (clear failure, not a silent fallback).
ALIASES = {
    "auto": "auto", "": "auto",
    "edit": "edit", "i2i": "edit", "editing": "edit",
    "t2i": "t2i", "gen": "t2i", "generate": "t2i", "txt2img": "t2i",
    "semseg": "semseg", "semantic": "semseg", "seg": "semseg",
    "refseg": "refseg", "referring": "refseg", "ref": "refseg",
    "instseg": "instseg", "instance": "instseg", "inst": "instseg",
    "depth": "depth",
    "normals": "normals", "normal": "normals",
    "raw": "raw",
}
PERCEPTION = ("semseg", "refseg", "instseg", "depth", "normals")
MASKS = ("semseg", "refseg", "instseg")  # colour-exact outputs -> NEAREST resize

# Colour list for semantic segmentation: the first six are the model card's own example colours.
PALETTE = [
    ("blue", (8, 4, 249)), ("magenta", (255, 8, 232)), ("green", (10, 255, 2)), ("cyan", (2, 204, 254)),
    ("red", (255, 0, 13)), ("yellow", (255, 249, 23)), ("orange", (255, 128, 0)), ("purple", (148, 0, 211)),
    ("aquamarine", (0, 255, 160)), ("pink", (255, 105, 180)), ("lime", (128, 255, 0)), ("azure", (0, 128, 255)),
    ("brown", (160, 82, 45)), ("gold", (255, 215, 0)), ("turquoise", (0, 206, 209)), ("crimson", (199, 21, 133)),
    ("olive", (154, 205, 50)), ("steel", (70, 130, 180)),
]

SEMSEG = ("[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, "
          "using this color mapping: {map} Color each pixel of a listed class with its color and leave everything "
          "else black. Output only the segmentation visualization image.")
REFSEG = ('[TASK: REFERRING_SEGMENTATION] Segment the region that best matches this description: "{d}". Paint the '
          "matching region red [220, 30, 30] and leave everything else black [0, 0, 0]. Output only the "
          "segmentation mask image.")
INSTSEG = ("[TASK: INSTANCE_SEGMENTATION] This image shows segmentation masks for {c} from the input image. "
           "The background is set to black color [0, 0, 0]. Each {c} instance is represented by a solid mask, "
           "and a different color is used for each mask. Instances are numbered by where their mask first "
           "appears in a left-to-right scan of the image (ties broken top to bottom). The 1st {c} is blue, "
           "the 2nd is lime, the 3rd is red, the 4th is sky blue, the 5th is dark green, the 6th is pink, "
           "the 7th is apricot, the 8th is mint, the 9th is wine, the 10th is magenta, the 11th is navy, "
           "the 12th is green, the 13th is yellow, the 14th is brown, the 15th is white, the 16th is olive, "
           "the 17th is lavender, the 18th is teal, the 19th is orange, the 20th is hot pink, the 21st is plum, "
           "the 22nd is gray, the 23rd is purple, the 24th is beige, the 25th is steel blue, the 26th is cyan, "
           "the 27th is royal blue, the 28th is sea green, the 29th is rust, the 30th is mustard, "
           "the 31st is mauve, the 32nd is maroon.")
DEPTH = ("Estimate a dense relative inverse-depth map for the input image. Produce exactly one output value "
         "for every input pixel at the same pixel location, with closer points brighter and farther points "
         "darker. Encode the result as neutral grayscale (R=G=B) using one consistent mapping over the "
         "complete image. Preserve object silhouettes, thin structures, occlusion boundaries, smooth surfaces, "
         "and small depth changes. Do not alter the scene or add texture, shading, color, text, labels, "
         "legends, borders, or objects; return only the registered depth map.")
NORMALS = ("Create a dense camera-space unit surface-normal map registered to the input. Use +x image-right, "
           "+y image-up, +z toward the camera and encode (x,y,z) as R=(1-x)/2, G=(1+y)/2, B=(1+z)/2. "
           "Preserve planes, curved surfaces, thin structures, and sharp orientation boundaries; "
           "return only the RGB normal map.")


def canonical_mode(mode):
    key = re.sub(r"[\s_\-]+", "", (mode or "").strip().lower())
    if key not in ALIASES:
        raise ValueError(f"OVB: unknown mode {mode!r}. Valid /slot1 modes: "
                         "auto, edit, t2i, semseg, refseg, instseg, depth, normals, raw.")
    return ALIASES[key]


def split_classes(text):
    parts = [p.strip().strip("\"'") for p in re.split(r"[,;\n|]+", text or "")]
    out = []
    for p in parts:
        if p and p.lower() not in [o.lower() for o in out]:
            out.append(p)
    return out


def build_prompt(mode, prompt, source_missing):
    """Returns (prompt, task, skip_source, lock_size)."""
    task = canonical_mode(mode)
    p = (prompt or "").strip()
    if task == "auto":
        task = "t2i" if source_missing else "edit"
    if task == "raw":
        return p, task, bool(source_missing), False
    if task == "t2i":
        if not p:
            raise ValueError("OVB: t2i needs a prompt describing the image.")
        return p, task, True, False
    if source_missing:
        raise ValueError(f"OVB: mode {task!r} needs a source image: reply to an image, attach one, or pass the "
                         "init image. Use /slot1:t2i (or leave /slot1 out) for text-to-image.")
    if task == "edit":
        if not p:
            raise ValueError("OVB: edit needs a prompt with the instruction (e.g. 'turn the horse into a zebra').")
        out = p if p.upper().startswith("[TASK:") else "[TASK: IMAGE_EDITING] " + p
    elif task == "semseg":
        classes = split_classes(p)
        if not classes:
            raise ValueError("OVB: semseg needs the class names in the prompt, comma separated "
                             "(e.g. 'road, sidewalk, sky, person, car').")
        if len(classes) > len(PALETTE):
            raise ValueError(f"OVB: semseg supports at most {len(PALETTE)} classes per run, got {len(classes)}.")
        cmap = "{" + ", ".join('"%s": [%d,%d,%d]' % ((c.replace('"', "'"),) + PALETTE[i][1])
                               for i, c in enumerate(classes)) + "}"
        out = SEMSEG.replace("{map}", cmap)
    elif task == "refseg":
        if not p:
            raise ValueError("OVB: refseg needs a description of the region in the prompt "
                             "(e.g. 'the skateboard under the skater').")
        out = REFSEG.replace("{d}", p.replace('"', "'"))
    elif task == "instseg":
        if not p:
            raise ValueError("OVB: instseg needs ONE class (noun phrase) in the prompt (e.g. 'car').")
        cls = split_classes(p)
        out = INSTSEG.replace("{c}", cls[0] if cls else p)
    elif task == "depth":
        out = DEPTH
    else:  # normals
        out = NORMALS
    return out, task, False, True


class OVBPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": ("STRING", {"default": "auto"}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "source_missing": ("BOOLEAN", {"default": True, "forceInput": True}),
        }, "optional": {
            "alt_missing": ("BOOLEAN", {"default": True, "forceInput": True}),
        }}

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING", "BOOLEAN")
    RETURN_NAMES = ("prompt", "skip_source", "task", "lock_size")
    FUNCTION = "run"
    CATEGORY = "OpenVisionBanana"

    def run(self, mode, prompt, source_missing, alt_missing=True):
        # a source exists if EITHER the init image or the alternative (/image1) arrived
        p, task, skip, lock = build_prompt(mode, prompt, bool(source_missing) and bool(alt_missing))
        print(f"[OVB] mode={task} skip_source={skip} lock_size={lock}\n[OVB] prompt: {p[:300]}")
        return (p, skip, task, lock)


class OVBRestoreSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "source": ("IMAGE",),
            "task": ("STRING", {"default": "edit", "forceInput": True}),
            "enabled": ("INT", {"default": 1, "min": 0, "max": 1}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "OpenVisionBanana"

    def run(self, image, source, task, enabled):
        if int(round(enabled)) < 1 or task not in PERCEPTION:
            return (image,)
        import torch.nn.functional as F
        h, w = int(source.shape[1]), int(source.shape[2])
        if image.shape[1] == h and image.shape[2] == w:
            return (image,)
        x = image.permute(0, 3, 1, 2)
        if task in MASKS:
            x = F.interpolate(x, size=(h, w), mode="nearest")
        else:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return (x.permute(0, 2, 3, 1).clamp(0, 1).contiguous(),)


NODE_CLASS_MAPPINGS = {"OVBPromptBuilder": OVBPromptBuilder, "OVBRestoreSize": OVBRestoreSize}
NODE_DISPLAY_NAME_MAPPINGS = {"OVBPromptBuilder": "OVB Prompt Builder", "OVBRestoreSize": "OVB Restore Size"}
