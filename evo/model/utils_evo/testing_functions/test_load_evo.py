from evo import Evo
import torch

device = 'cuda:0'

evo_model = Evo('evo-1-131k-base')
model, tokenizer = evo_model.model, evo_model.tokenizer
model.to(device)
model.eval()

sequence = 'ACGT'
input_ids = torch.tensor(
    tokenizer.tokenize(sequence),
    dtype=torch.int,
).to(device).unsqueeze(0)
logits, _ = model(input_ids) # (batch, length, vocab)

print('Logits: ', logits)
print('Shape (batch, length, vocab): ', logits.shape)
