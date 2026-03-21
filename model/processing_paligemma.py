from typing import List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch


IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def resize(image: Image, size: Tuple[int, int], resample: Image.Resampling=None, reducing_gap: Optional[int]=None) -> np.ndarray :
    height, width=size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def rescale(image: np.ndarray, scale: float, dtype: np.dtype=np.float32) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(image: np.ndarray,
              mean: Union[float,Iterable[float]],
              std: Union[float, Iterable[float]],
)->np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def process_images(
    images: List[Image.Image],
    size: Tuple[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
)->List[np.ndarray]:
    height, width = size[0], size[1]
    images = [resize(image=image, size=(height, width), resample=resample) for image in images]
    # covert each image to a numpy array
    images = [np.array(image) for image in images]
    # rescale the pixel values to be in range [0, 1]
    images = [rescale(image, scale = rescale_factor) for image in images]
    # Normalize the images
    images = [normalize(image, mean = image_mean, std = image_std) for image in images]
    # move the channel dimension to the first dimension
    images = [image.transpose(2, 0, 1) for image in images]
    return images

class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int, ignore_index: int = -100):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        self.ignore_index = ignore_index

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        # These tokens are used for object detection(bounding box)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        # These tokens are used for object segmentation
        EXTRA_TOKENS += [f"<seg{i:03}>" for i in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def _build_prompt_strings(self, text: List[str]) -> List[str]:
        return [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

    def _append_suffix(self, prompt: str, suffix: str) -> str:
        suffix = "" if suffix is None else suffix
        eos_token = self.tokenizer.eos_token or ""
        if eos_token and suffix.endswith(eos_token):
            return prompt + suffix
        return prompt + suffix + eos_token

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        suffix: Optional[List[str]] = None,
        padding: str = "longest",
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_labels: bool = False,
    ) -> dict:
        assert len(images) == len(text), f"Receive {len(images)} images for {len(text)} prompts."
        if suffix is not None:
            assert len(suffix) == len(text), f"Receive {len(suffix)} suffixes for {len(text)} prompts."
        if max_length is not None and max_length <= self.image_seq_length:
            raise ValueError(
                f"`max_length` must be greater than the image token count ({self.image_seq_length})."
            )

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 /255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD,
        )
        # convert the list pf numpy arrays to a single numpy array with shape [Batch_size, Channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis = 0)
        # convert the numpy array to a pytorch tensor
        pixel_values = torch.tensor(pixel_values)

        input_strings = self._build_prompt_strings(text)
        model_strings = input_strings
        if suffix is not None:
            model_strings = [
                self._append_suffix(prompt, target)
                for prompt, target in zip(input_strings, suffix)
            ]
        # return the input_ids and attention_mask as Pytorch tensors
        inputs = self.tokenizer(
            model_strings,
            return_tensors = "pt",
            padding = padding,
            truncation = truncation,
            max_length = max_length,
        )
        image_token_counts = (inputs["input_ids"] == self.image_token_id).sum(dim=-1)
        if not torch.all(image_token_counts == self.image_seq_length):
            raise ValueError(
                "Image token count does not match the expected patch count. "
                "Check tokenizer special tokens or increase `max_length`."
            )

        return_data = {"pixel_values": pixel_values, **inputs}

        if return_labels:
            prompt_inputs = self.tokenizer(
                input_strings,
                padding = False,
                truncation = truncation,
                max_length = max_length,
            )
            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = self.ignore_index
            for index, prompt_input_ids in enumerate(prompt_inputs["input_ids"]):
                prompt_length = min(len(prompt_input_ids), labels.shape[1])
                labels[index, :prompt_length] = self.ignore_index

            return_data["labels"] = labels

        return return_data
