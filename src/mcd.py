from typing import Optional, Tuple
import numpy as np
import librosa


class Stft:
    """STFT module."""

    def __init__(self):
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = 160
        self.center = True
        self.onesided = True
        self.normalized = False
        self.window = librosa.filters.get_window("hann", self.win_length)

    def __call__(
        self, input: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """STFT forward function.
        Args:
            input: (Batch, Nsamples)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2)
        """
        stft_kwargs = dict(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            window=self.window,
            pad_mode="reflect",
        )
        output = []
        # iterate over istances in a batch
        for i, instance in enumerate(input):
            stft = librosa.stft(input[i], **stft_kwargs)
            output.append(np.array(np.stack([stft.real, stft.imag], -1)))
        output = np.vstack(output).reshape(len(output), *output[0].shape)

        if not self.onesided:
            len_conj = self.n_fft - output.shape[1]
            conj = output[:, 1 : 1 + len_conj].flip(1)
            conj[:, :, :, -1].data *= -1
            output = np.concatenate([output, conj], 1)
        if self.normalized:
            output = output * (stft_kwargs["window"].shape[0] ** (-0.5))

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        print(f"Output shape: {output.shape}")
        output = output.transpose(0, 2, 1, 3)

        # create complex array
        output = output[..., 0] + output[..., 1] * 1j
        return output


class LogMel:
    """Convert STFT to fbank feats
    """

    def __init__(self):
        fmin = 80
        fmax = 16000 / 2
        _mel_options = dict(
            sr=16000,
            n_fft=1024,
            n_mels=80,
            fmin=fmin,
            fmax=fmax,
            htk=False,
        )
        self.mel_options = _mel_options
        melmat = librosa.filters.mel(**_mel_options)
        self.melmat = melmat.T

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def __call__(
        self,
        speech: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = np.matmul(speech, self.melmat)
        mel_feat = np.clip(mel_feat, 1e-10, float("inf"))

        logmel_feat = np.log(mel_feat)

        return logmel_feat


stft = Stft()
logmel = LogMel()


def compute_mcd(original, synthesized):
    """Compute Mel Cepstrum Distortion (MCD) between original and synthesized Mel Cepstral Coefficients (MCCs).
    Args:
        original (np.ndarray): Original Mel Cepstral Coefficients (MCCs) of shape (num_frames, num_cepstral_coefficients).
        synthesized (np.ndarray): Synthesized Mel Cepstral Coefficients (MCCs) of shape (num_frames, num_cepstral_coefficients).
    """
    orig_stft = stft(original.detach().cpu().numpy())
    synth_stft = stft(synthesized.detach().cpu().numpy()[None])

    orig_power = orig_stft.real**2 + orig_stft.imag**2
    synth_power = synth_stft.real**2 + synth_stft.imag**2

    orig_logmel = logmel(orig_stft)
    synth_logmel = logmel(synth_stft)

    l1_loss = np.abs(orig_logmel - synth_logmel)
    coef = 10.0 / np.log(np.array([10])) \
                         * np.sqrt(np.array([2]))
    mcd = coef * np.sum(l1_loss, 2)
    return np.mean(mcd)
    
