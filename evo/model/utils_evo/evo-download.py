from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model name
model_name = 'togethercomputer/evo-1-8k-base'

# Download and save the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("togethercomputer/evo-1-8k-base", trust_remote_code=True) 
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally
model.save_pretrained('./local_evo_model')
tokenizer.save_pretrained('./local_evo_model')
