from tt.model import Seq2SeqTransformer
import torch

model = Seq2SeqTransformer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#random inputs
src = torch.randint(0, 10000, (10, 32)).to(device)
trg = torch.randint(0, 10000, (20, 32)).to(device)

src_mask = model.generate_square_subsequent_mask(
    src.size(0)
).to(device)

tgt_mask = model.generate_square_subsequent_mask(
    trg.size(0)
).to(device)

src_padding_mask = (src == 0)
tgt_padding_mask = (trg == 0)

memory_key_padding_mask = src_padding_mask.clone()

#forward pass
outs = model(
    src,
    trg,
    src_mask,
    tgt_mask,
    src_padding_mask,
    tgt_padding_mask,
    memory_key_padding_mask
)