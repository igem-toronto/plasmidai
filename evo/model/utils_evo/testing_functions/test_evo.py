import torch
from evo import Evo

# Initialize the device
device = "cuda:0"

# Load the model and tokenizer
evo_model = Evo("evo-1-131k-base")
model, tokenizer = evo_model.model, evo_model.tokenizer
model.to(device)
model.eval()

# Define the sequence and convert it to input ids
sequence = "ACGTAI"
input_ids = (
    torch.tensor(
        tokenizer.tokenize(sequence),
        dtype=torch.int,
    )
    .to(device)
    .unsqueeze(0)
)
output = model(input_ids)  # (batch, length, vocab)
print(type(output))
logits, _ = output
# Print the logits and their shape
print("Logits: ", logits)

# B C L
print("Shape (batch, length, vocab): ", logits.shape)

# # Print the model's layers and their weights
# print("Model's layers and weights:")
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")  # Printing the first 2 values for brevity

# # Replace the last layer with a forward (linear) layer
# # Assuming the last layer is a fully connected layer and we replace it with a similar layer
# # Find the dimension of the output of the penultimate layer
# output_dim = model.config.hidden_size  # or the dimension of the layer before the last layer
# vocab_size = logits.shape[-1]  # The size of the vocabulary

# # Create a new linear layer
# new_last_layer = torch.nn.Linear(output_dim, vocab_size).to(device)

# # Replace the last layer in the model
# # This step depends on the specific architecture of your model
# # Here's a general approach, it might need adjustments based on the actual model architecture
# model.lm_head = new_last_layer

# # Check the updated model
# logits, _ = model(input_ids)  # (batch, length, vocab)
# print('New Logits: ', logits)
# print('Shape (batch, length, vocab): ', logits.shape)

# # Print the evo_model (or its structure)
print(model)
print(
    type(model)
)  # this tells me the model is a class, that the class is StripedHyena... does that mean I can change the attributes of the class, notably change the forwrad functino
# print(model[-1])
# print(type(model[-1]))
print(vars(model))  # returns all the attributes of a mdoel

# model.
# maybe try this:
# https://discuss.pytorch.org/t/how-to-replace-a-layer-or-module-in-a-pretrained-network/60068
print(model.__dict__)
