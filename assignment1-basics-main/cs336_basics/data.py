import numpy.typing as npt
import torch
def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack(
        [torch.tensor(dataset[s : s + context_length], dtype=torch.long) for s in starts]
    )
    y = torch.stack(
        [torch.tensor(dataset[s + 1 : s + context_length + 1], dtype=torch.long) for s in starts]
    )
    return x.to(device), y.to(device)