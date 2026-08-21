"""WWAI expose/marker nodes for ComfyUI.

Every node here is a pure pass-through: it forwards its `value` input to
whatever it is wired to, the same shape as ComfyUI's own Reroute/Primitive
nodes. The `name` and `description` widgets carry no runtime behavior — they
exist so an external reader (WWAI) can later discover, by static inspection
of the saved workflow JSON, which sockets a workflow author intended to
expose as controllable inputs or as the finished result.
"""


class AnyType(str):
    """A type string that reports equal to any other type.

    ComfyUI has no built-in VIDEO or MESH type, and third-party node packs
    each define their own. Using this wildcard for the video/mesh sockets
    lets a WWAIExpose* node wire to whatever concrete type the rest of the
    graph uses, without depending on any specific third-party package.
    """

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


ANY = AnyType("*")


def _name_and_description_inputs():
    return {
        "name": ("STRING", {"default": ""}),
        "description": ("STRING", {"multiline": True, "default": ""}),
    }


class WWAIExposeText:
    CATEGORY = "WWAI/Expose"
    FUNCTION = "expose"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"multiline": True, "default": ""}),
                **_name_and_description_inputs(),
            }
        }

    def expose(self, value, name, description):
        return (value,)


class WWAIExposeInt:
    CATEGORY = "WWAI/Expose"
    FUNCTION = "expose"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -0x7FFFFFFF, "max": 0x7FFFFFFF}),
                **_name_and_description_inputs(),
            }
        }

    def expose(self, value, name, description):
        return (value,)


class WWAIExposeFloat:
    CATEGORY = "WWAI/Expose"
    FUNCTION = "expose"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "step": 0.01}),
                **_name_and_description_inputs(),
            }
        }

    def expose(self, value, name, description):
        return (value,)


class WWAIExposeImage:
    CATEGORY = "WWAI/Expose"
    FUNCTION = "expose"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("IMAGE",),
                **_name_and_description_inputs(),
            }
        }

    def expose(self, value, name, description):
        return (value,)


class WWAIExposeVideo:
    CATEGORY = "WWAI/Expose"
    FUNCTION = "expose"
    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY,),
                **_name_and_description_inputs(),
            }
        }

    def expose(self, value, name, description):
        return (value,)


class WWAIExposeMesh:
    CATEGORY = "WWAI/Expose"
    FUNCTION = "expose"
    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY,),
                **_name_and_description_inputs(),
            }
        }

    def expose(self, value, name, description):
        return (value,)


class WWAIExposeOutputImage:
    CATEGORY = "WWAI/Expose Output"
    FUNCTION = "expose"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("IMAGE",),
                **_name_and_description_inputs(),
            }
        }

    def expose(self, value, name, description):
        return (value,)


class WWAIExposeOutputVideo:
    CATEGORY = "WWAI/Expose Output"
    FUNCTION = "expose"
    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY,),
                **_name_and_description_inputs(),
            }
        }

    def expose(self, value, name, description):
        return (value,)


class WWAIExposeOutputMesh:
    CATEGORY = "WWAI/Expose Output"
    FUNCTION = "expose"
    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY,),
                **_name_and_description_inputs(),
            }
        }

    def expose(self, value, name, description):
        return (value,)


NODE_CLASS_MAPPINGS = {
    "WWAIExposeText": WWAIExposeText,
    "WWAIExposeInt": WWAIExposeInt,
    "WWAIExposeFloat": WWAIExposeFloat,
    "WWAIExposeImage": WWAIExposeImage,
    "WWAIExposeVideo": WWAIExposeVideo,
    "WWAIExposeMesh": WWAIExposeMesh,
    "WWAIExposeOutputImage": WWAIExposeOutputImage,
    "WWAIExposeOutputVideo": WWAIExposeOutputVideo,
    "WWAIExposeOutputMesh": WWAIExposeOutputMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WWAIExposeText": "WWAI Expose Text",
    "WWAIExposeInt": "WWAI Expose Int",
    "WWAIExposeFloat": "WWAI Expose Float",
    "WWAIExposeImage": "WWAI Expose Image",
    "WWAIExposeVideo": "WWAI Expose Video",
    "WWAIExposeMesh": "WWAI Expose Mesh",
    "WWAIExposeOutputImage": "WWAI Expose Output Image",
    "WWAIExposeOutputVideo": "WWAI Expose Output Video",
    "WWAIExposeOutputMesh": "WWAI Expose Output Mesh",
}
