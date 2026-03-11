from Bio import SeqIO

file1 = "85-cd-hit-epitope.fasta"
file2 = "DatasetB_Negative_epitopes_16.fasta"
output = "DatasetB_Negative_epitopes_16_filtered.fasta"

seqs_file1 = set(str(record.seq) for record in SeqIO.parse(file1, "fasta"))

with open(output, "w") as out:
    for record in SeqIO.parse(file2, "fasta"):
        if str(record.seq) not in seqs_file1:
            SeqIO.write(record, out, "fasta")

print("Duplicados exactos eliminados.")
