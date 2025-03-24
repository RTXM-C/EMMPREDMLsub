import torch
import esm
from Bio import SeqIO
from predict import main_predict


def process_fasta(file_path):
    sequences = []
    with open(file_path, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            seq = str(record.seq)
            if len(seq) >= 70:
                merged_seq = seq[:35] + seq[-35:]
                sequences.append(merged_seq)
            else:
                print(f"Warning: Sequence {record.id} is shorter than 70 nucleotides and will be skipped.")
    return sequences


def esm_feature_small(sequences):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = [("mrna", seq) for seq in sequences]
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    batch_size = 100
    features = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        with torch.no_grad():
            results = model(batch_tokens.to(device), repr_layers=[30])
        features.append(results["representations"][30])
    return torch.cat(features, dim=0).cpu().numpy()


def location_coding(pros):
    pre = (pros > 0.5).astype(int)
    locs = ["".join(map(str, row)) for row in pre]
    return locs


file_path = "sample.txt"
sequences = process_fasta(file_path)
features = esm_feature_small(sequences)
pros = main_predict(features)
print(location_coding(pros))
