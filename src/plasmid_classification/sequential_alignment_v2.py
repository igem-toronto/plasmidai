from Bio import SeqIO, pairwise2
from Bio.pairwise2 import format_alignment
import csv
import concurrent.futures
from tqdm import tqdm

def align_sequences(seq1, seq2, match=2, mismatch=-1, gap_open=-2, gap_extend=-0.5):
    """Align two sequences and return the best normalized alignment score."""
    alignments = pairwise2.align.localxx(seq1, seq2)
    best_alignment = alignments[0]
    alignment_length = best_alignment[4]  # Length of the alignment
    score = best_alignment[2]
    normalized_score = score / alignment_length  # Normalize by alignment length
    return normalized_score

def calculate_gc_content(seq):
    g = seq.count("G")
    c = seq.count("C")
    return 100 * (g + c) / len(seq)

def process_plasmid(plasmid, elements, threshold):
    plasmid_scores = {'sequence_name': plasmid.id}
    total_score = 0
    
    # Calculate GC content
    gc_content = calculate_gc_content(str(plasmid.seq))
    plasmid_scores['gc_content'] = gc_content
    
    for element in elements:
        normalized_score = align_sequences(str(plasmid.seq), str(element.seq))
        plasmid_scores[element.id] = normalized_score
        
        if normalized_score >= threshold:
            total_score += normalized_score
    
    plasmid_scores['total_score'] = total_score
    return plasmid_scores

def sequential_alignment(unknown_plasmid_file, elements_file, threshold=0.8):
    unknown_plasmids = list(SeqIO.parse(unknown_plasmid_file, "fasta"))
    elements = list(SeqIO.parse(elements_file, "fasta"))

    results = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_plasmid, plasmid, elements, threshold): plasmid for plasmid in unknown_plasmids}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing plasmids"):
            results.append(future.result())

    # Sort the results based on total_score, priority to P2 and GC content
    results = sorted(results, key=lambda x: (x['total_score'], x.get('P2', 0), x['gc_content']), reverse=True)
    
    return results

def save_to_csv(results, output_file='plasmid_alignment_results.csv'):
    if results:
        fieldnames = ['sequence_name', 'gc_content'] + [key for key in results[0] if key not in ('sequence_name', 'gc_content', 'total_score')] + ['total_score']

        with open(output_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

if __name__ == "__main__":
    unknown_plasmid_file = "new_seqs_2024-07-26.fasta"
    elements_file = "elements.fasta"
    
    results = sequential_alignment(unknown_plasmid_file, elements_file, threshold=0.8)
    output_file_name = f"{unknown_plasmid_file.rsplit('.', 1)[0]}_local_alignment_results_normalized.csv"
    save_to_csv(results, output_file=output_file_name)
    
    print(f"Alignment results saved to '{output_file_name}'.")
