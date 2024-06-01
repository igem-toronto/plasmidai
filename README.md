# plasmid-ai

---

## Environment Setup

Download the dependencies: 

```
conda env create -f environment.yml
```

Download the plasmid sequences:

```bash
cd data 
gdown "1iIsat00ST5vK-06BUstuTbJkfWKpV2lE" 
gzip -d 240212_plasmid_seq_54646.fasta.gz
mv 240212_plasmid_seq_54646.fasta plasmids.fasta
```

## Training  

### LLM 

Example LLM training command (single GPU):

```bash 
python -m src.experimental.llm.train \
    --accelerator=gpu --devices=1 \
    --precision=bf16-mixed \
    --batch_size=32 --num_workers=8 \
    --enable_fused_add_norm \
    --enable_wandb \
    --enable_checkpoint
```