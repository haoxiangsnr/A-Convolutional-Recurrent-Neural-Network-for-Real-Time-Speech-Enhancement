import librosa
import torch
from tqdm import tqdm

from inferencer.base_inferencer import BaseInferencer
from inferencer.inferencer_full_band import full_band_no_truncation


@torch.no_grad()
def inference_wrapper(
        dataloader,
        model,
        device,
        inference_args,
        enhanced_dir
):
    for noisy, name in tqdm(dataloader, desc="Inference"):
        assert len(name) == 1, "The batch size of inference stage must 1."
        name = name[0]

        noisy = noisy.numpy().reshape(-1)

        if inference_args["inference_type"] == "full_band_no_truncation":
            noisy, enhanced = full_band_no_truncation(model, device, inference_args, noisy)
        else:
            raise NotImplementedError(f"Not implemented Inferencer type: {inference_args['inference_type']}")

        print(enhanced_dir / f"{name}.wav")
        librosa.output.write_wav(enhanced_dir / f"{name}.wav", enhanced, sr=16000)


class Inferencer(BaseInferencer):
    def __init__(self, config, checkpoint_path, output_dir):
        super(Inferencer, self).__init__(config, checkpoint_path, output_dir)

    @torch.no_grad()
    def inference(self):
        inference_wrapper(
            dataloader=self.dataloader,
            model=self.model,
            device=self.device,
            inference_args=self.inference_config,
            enhanced_dir=self.enhanced_dir
        )
