import time
import math

import torch

from tqdm import trange

from model import GPTConfig, GPT
from dataloader import DataLoaderLite

torch.set_float32_matmul_precision("high")

torch.manual_seed(1337)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(1337)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"using device: {device}")

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

optimizer = model.configure_optimizer(
    weight_decay=0.1, learning_rate=6e-4, device=device
)
scaler = torch.amp.grad_scaler.GradScaler(device)

total_batch_size = 524288
B = 1
T = 1024
assert total_batch_size % (B * T) == 0

grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B, T, "train")
val_loader = DataLoaderLite(B, T, "val")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073


def get_lr(it: int):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


for step in range(max_steps):
    t0 = time.time()

    if step % 100 == 0:
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in trange(val_loss_steps, desc="validation", leave=False):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)

                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(x, y)

                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        print(f"validation loss: {val_loss_accum.item():.4f}")

    model.train()

    optimizer.zero_grad(set_to_none=True)

    loss_accum = 0.0

    for micro_step in trange(grad_accum_steps, desc=f"step {step + 1}", leave=False):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)

        loss /= grad_accum_steps
        loss_accum += loss.detach()

        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    scaler.step(optimizer)
    scaler.update()

    # torch.cuda.synchronize()
    t1 = time.time()

    dt = t1 - t0

    tok_processed = train_loader.B * train_loader.T * grad_accum_steps
    tok_rate = tok_processed / dt

    print(
        f"step: {step + 1}/{max_steps} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | time: {dt:.2f}s | tok/s: {tok_rate:.2f}"
    )
