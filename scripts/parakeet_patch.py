import math
import mlx.core as mx
import mlx.nn as nn
import parakeet_mlx.conformer
from parakeet_mlx.conformer import ConformerArgs

def apply_patch():
    """
    Monkey-patch parakeet_mlx.conformer.DwStridingSubsampling to support causal_downsampling.
    
    This is required because the original model uses causal_downsampling=True,
    but parakeet-mlx v0.5.0 only supports False.
    """
    print("[INFO] Applying runtime patch to parakeet_mlx for causal_downsampling support...")
    parakeet_mlx.conformer.DwStridingSubsampling = PatchedDwStridingSubsampling

class PatchedDwStridingSubsampling(nn.Module):
    def __init__(self, args: ConformerArgs):
        super().__init__()

        assert (
            args.subsampling_factor > 0
            and (args.subsampling_factor & (args.subsampling_factor - 1)) == 0
        )
        self.subsampling_conv_chunking_factor = args.subsampling_conv_chunking_factor
        self._conv_channels = args.subsampling_conv_channels
        self._sampling_num = int(math.log(args.subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._padding = (self._kernel_size - 1) // 2
        
        # ADDED: Support for causal downsampling
        self.causal_downsampling = getattr(args, 'causal_downsampling', False)

        in_channels = 1
        final_freq_dim = args.feat_in
        for i in range(self._sampling_num):
            # FIXED: If causal, first layer freq padding is 2 (symmetric equivalent), others 1.
            pad_f = 2 if (self.causal_downsampling and i == 0) else self._padding
            
            final_freq_dim = (
                math.floor(
                    (final_freq_dim + 2 * pad_f - self._kernel_size)
                    / self._stride
                )
                + 1
            )
            if final_freq_dim < 1:
                raise ValueError("Non-positive final frequency dimension!")

        # FIXED: If causal, we use manual padding in conv_forward, so set conv padding to 0 here
        conv_padding = 0 if self.causal_downsampling else self._padding

        self.conv = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self._conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=conv_padding,
            ),
            nn.ReLU(),
        ]
        in_channels = self._conv_channels

        for _ in range(self._sampling_num - 1):
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=conv_padding,
                    groups=in_channels,
                )
            )
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self._conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            self.conv.append(nn.ReLU())

        self.out = nn.Linear(self._conv_channels * final_freq_dim, args.d_model)

    def conv_forward(self, x: mx.array) -> mx.array:
        x = x.transpose((0, 2, 3, 1)) # (N, Time, Freq, C)
        
        # Track which strided conv layer we are on
        strided_conv_idx = 0
        
        for i, layer in enumerate(self.conv):
            # Check if this is one of the strided/kernel=3 convolutions
            # The structure is [Conv3x3, ReLU, Conv3x3, Conv1x1, ReLU, Conv3x3, Conv1x1, ReLU ...]
            # Indices of Conv3x3: 0, 2, 5, 8...
            is_strided_conv = (i == 0) or (i > 0 and (i - 2) % 3 == 0)
            
            if self.causal_downsampling and is_strided_conv:
                # Time padding: Causal (2 left, 0 right)
                pad_t = (2, 0)
                
                # Freq padding: 
                # Layer 0: 2 symmetric (2 left, 2 right)
                # Others: 1 symmetric (1 left, 1 right)
                pad_f = (2, 2) if strided_conv_idx == 0 else (1, 1)
                
                x = mx.pad(x, ((0, 0), pad_t, pad_f, (0, 0)))
                strided_conv_idx += 1
                
            x = layer(x)
            
        return x.transpose((0, 3, 1, 2))

    def conv_split_by_batch(self, x: mx.array) -> tuple[mx.array, bool]:
        # Helper required by parakeet-mlx, copied from original
        b = x.shape[0]
        if b == 1:
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(x.size / x_ceil, 2))
            cf: int = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:
            return x, False

        return mx.concat(
            [self.conv_forward(chunk) for chunk in mx.split(x, new_batch_size, 0)]
        ), True

    def __call__(self, x: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        # Recalculate lengths - this logic remains mostly same but we need to ensure correct loop
        for _ in range(self._sampling_num):
            lengths = (
                mx.floor(
                    (lengths + 2 * self._padding - self._kernel_size) / self._stride
                )
                + 1.0
            )
        lengths = lengths.astype(mx.int32)

        x = mx.expand_dims(x, axis=1)

        if self.subsampling_conv_chunking_factor != -1:
            if self.subsampling_conv_chunking_factor == 1:
                x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
                need_to_split = x.size > x_ceil
            else:
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:
                    x = self.conv_forward(x)  # try anyways
            else:
                x = self.conv_forward(x)
        else:
            x = self.conv_forward(x)

        b, t, f, c = x.shape
        x = x.reshape(b, t, f * c)
        x = self.out(x)
        return x, lengths
