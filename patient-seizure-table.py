import csv
import sys


exp = sys.argv[1]
print('Patient,FPs per Hour,Latency(s),Sensitivity')
total_fp = 0
total_latency = 0
total_sensitivity = 0
for pp in range(1, 35):
    fn = (
        '{}/pt{}/results/val_seizure_results_smoothed.csv'
    ).format(exp, pp)
    with open(fn) as f:
        csvreader = csv.DictReader(f)
        results = []
        for row in csvreader:
            results.append(row)
    total_fp += float(results[-1]['fps_per_hour'])
    total_latency += float(results[-1]['latency_time'])
    total_sensitivity += float(results[-1]['sensitivity'])
    outstr = 'Patient {},{},{},{}'.format(pp, results[-1]['fps_per_hour'],
                                          results[-1]['latency_time'],
                                          results[-1]['sensitivity'])
    print(outstr)

# Compute Averages
avg_fp = total_fp / 34
avg_latency = total_latency / 34
avg_sensitivity = total_sensitivity / 34
outstr = '{},{},{},{}'.format('Average', avg_fp, avg_latency,
                              avg_sensitivity)
print(outstr)
