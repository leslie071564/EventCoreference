import sys

def aggregate_result(result_file):
    precs, recalls, f1s = [], [], []
    for line in open(result_file, 'r'):
        if line.startswith('conll'):
            _, p, r, f1 = line.rstrip().split()
            precs.append(float(p))
            recalls.append(float(r))
            f1s.append(float(f1))

    best_index = f1s.index(max(f1s))
    with open(result_file, 'w+') as F:
        F.write("##### average of document CoNLL f1:\n")
        F.write('precision: %.2f\n' % (sum(precs) / len(precs)))
        F.write('recall   : %.2f\n' % (sum(recalls) / len(recalls)))
        F.write('F1       : %.2f\n' % (sum(f1s) / len(f1s)))
        F.write('best performance: model_%s\n' % (best_index + 1))

if __name__ == "__main__":
    result_file = sys.argv[1]
    aggregate_result(result_file)
