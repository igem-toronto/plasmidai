from generate import reference_evo_generate


#~ use what is provided by evo already
# def generate_sequences(model, start_sequence, tokenizer, max_length=100, num_sequences=10):
#     model.eval()
#     sequences = []
#     with torch.no_grad():
#         for _ in range(num_sequences):
#             input_seq = tokenizer(start_sequence)
#             input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
#             generated_seq = start_sequence
#             for _ in range(max_length):
#                 output = model(input_tensor)
#                 next_token = torch.argmax(output, dim=-1).item()
#                 generated_seq += tokenizer.decode([next_token])
#                 input_tensor = torch.cat((input_tensor, torch.tensor([[next_token]], dtype=torch.long)), dim=-1)
#                 if next_token == tokenizer.eos_token_id:
#                     break
#             sequences.append(generated_seq)
#     return sequences

# # Example usage
# # Assuming `pretrained_model` is an instance of a pre-trained Evo model
# # and `tokenizer` is an instance of a tokenizer class

# # Load your dataset
# sequences = ["ATGCGT", "TTAGGC", "CGATCG"]  # Example sequences
# tokenizer = ...  # Initialize your tokenizer
# train_dataset = PlasmidDataset(sequences, tokenizer)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # Initialize the EvoMamba model
# evo_mamba = EvoMamba(pretrained_model, mamba_units=128)

# # Train the model
# train_model(evo_mamba, train_loader, val_loader=None, epochs=10, learning_rate=0.001)

# # Generate new sequences
# start_sequence = "ATG"
# generated_sequences = generate_sequences(evo_mamba, start_sequence, tokenizer)
# for seq in generated_sequences:
#     print(seq)
