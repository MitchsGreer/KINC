import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
import scipy.interpolate
import scipy.stats
import seaborn as sns
import sklearn.cluster
import sklearn.mixture
import sys



def load_cmx(filename, num_genes, num_clusters):
	netlist = pd.read_csv(args.INPUT, sep="\t", header=None)
	cmx = np.zeros((num_genes * num_clusters, num_genes * num_clusters), dtype=np.float32)

	for idx in range(len(netlist.index)):
		i = netlist.iloc[idx, 0]
		j = netlist.iloc[idx, 1]
		k = netlist.iloc[idx, 2]
		r = netlist.iloc[idx, 9]

		cmx[i * num_clusters + k, j * num_clusters + k] = r
		cmx[j * num_clusters + k, i * num_clusters + k] = r

	return cmx



def powerlaw(args):
	# load correlation matrix
	S = load_cmx(args.INPUT, args.NUM_GENES, args.MAX_CLUSTERS)

	# iterate until network is sufficiently scale-free
	threshold = args.TSTART

	while True:
		# compute thresholded adjacency matrix
		A = (abs(S) >= threshold)

		# compute degree of each node
		for i in range(A.shape[0]):
			A[i, i] = 0

		degrees = np.array([sum(A[i]) for i in range(A.shape[0])])

		# compute degree distribution
		bins = max(5, degrees.max())
		hist, _ = np.histogram(degrees, bins=bins, range=(1, bins))
		bin_edges = range(1, len(hist) + 1)

		# modify histogram values to work with loglog plot
		hist += 1

		# plot degree distribution
		if args.VISUALIZE:
			plt.subplots(1, 2, figsize=(10, 5))
			plt.subplot(121)
			plt.title("Degree Distribution")
			plt.plot(bin_edges, hist, "ko")
			plt.subplot(122)
			plt.title("Degree Distribution (log-log)")
			plt.loglog(bin_edges, hist, "ko")
			plt.savefig("powerlaw_%03d.png" % (int(threshold * 1000)))
			plt.close()

		# compute correlation
		x = np.log(bin_edges)
		y = np.log(hist)

		r, p = scipy.stats.pearsonr(x, y)

		# output results of threshold test
		print("%g\t%g\t%g" % (threshold, r, p))

		# break if power law is satisfied
		if r < 0 and p < 1e-20:
			break

		# decrement threshold and fail if minimum threshold is reached
		threshold -= args.TSTEP
		if threshold < args.TSTOP:
			print("error: could not find an adequate threshold above stopping threshold")
			sys.exit(0)

	return threshold



def compute_pruned_matrix(S, threshold):
	S_pruned = np.copy(S)
	S_pruned[abs(S) < threshold] = 0
	S_pruned = S_pruned[~np.all(S_pruned == 0, axis=1)]
	S_pruned = S_pruned[:, ~np.all(S_pruned == 0, axis=0)]

	return S_pruned



def compute_unique(values):
	unique = []
	
	for i in range(len(values)):
		if len(unique) == 0 or abs(values[i] - unique[-1]) > 1e-6:
			unique.append(values[i])

	return unique



def compute_spline(values, pace):
	# extract values for spline based on pace
	x = values[::pace]
	y = np.linspace(0, 1, len(x))

	# compute spline
	spl = scipy.interpolate.splrep(x, y)

	# extract interpolated eigenvalues from spline
	spline_values = scipy.interpolate.splev(values, spl)

	return spline_values



def compute_spacings(values):
	spacings = np.empty(len(values) - 1)
	
	for i in range(len(spacings)):
		spacings[i] = (values[i + 1] - values[i]) * len(values)

	return spacings



def compute_chi_square_helper(values):
	# compute eigenvalue spacings
	spacings = compute_spacings(values)

	# compute nearest-neighbor spacing distribution
	hist_min = 0.0
	hist_max = 3.0
	num_bins = 60
	bin_width = (hist_max - hist_min) / num_bins

	hist, _ = np.histogram(spacings, num_bins, (hist_min, hist_max))
	
	# compote chi-square value from nnsd
	chi = 0
	
	for i in range(len(hist)):
		# compute O_i, the number of elements in bin i
		O_i = hist[i]

		# compute E_i, the expected value of Poisson distribution for bin i
		E_i = (math.exp(-i * bin_width) - math.exp(-(i + 1) * bin_width)) * len(values)

		# update chi-square value based on difference between O_i and E_i
		chi += (O_i - E_i) * (O_i - E_i) / E_i

	return chi



def compute_chi_square(eigens, spline=True):
	# make sure there are enough eigenvalues
	if len(eigens) < 50:
		return -1

	# use spline interpolation if specified
	if spline:
		# perform several chi-square tests with spline interpolation by varying the pace
		chi = 0
		num_tests = 0

		for pace in range(10, 41):
			# make sure there are enough eigenvalues for pace
			if len(eigens) / pace < 5:
				break

			# compute spline-interpolated eigenvalues
			eigens = compute_spline(eigens, pace)

			# compute chi-squared value
			chi_pace = compute_chi_square_helper(eigens)

			print("pace: %d, chi-squared: %g" % (pace, chi_pace))

			# compute chi-squared value
			chi += chi_pace
			num_tests += 1

		# return average of chi-square tests
		return chi / num_tests

	# perform a single chi-squared test without spline interpolation
	else:
		return compute_chi_square_helper(eigens)



def rmt(args):
	# load correlation matrix
	S = load_cmx(args.INPUT, args.NUM_GENES, args.MAX_CLUSTERS)

	# iterate until chi value goes below 99.607 then above 200
	final_threshold = 0
	final_chi = float("inf")
	max_chi = -float("inf")
	threshold = args.TSTART

	while max_chi < 200:
		# compute pruned matrix
		S_pruned = compute_pruned_matrix(S, threshold)

		# make sure pruned matrix is not empty
		chi = -1

		if S_pruned.shape[0] > 0:
			# compute eigenvalues of pruned matrix
			eigens, _ = np.linalg.eigh(S_pruned)

			print("eigenvalues: %d" % len(eigens))

			# compute unique eigenvalues
			eigens = compute_unique(eigens)

			print("unique eigenvalues: %d" % len(eigens))

			# compute chi-square value from NNSD of eigenvalues
			chi = compute_chi_square(eigens, spline=args.SPLINE)

		# make sure chi-square test succeeded
		if chi != -1:
			# plot eigenvalue distribution
			if args.VISUALIZE:
				plt.subplots(1, 2, figsize=(10, 5))
				plt.subplot(121)
				plt.title("Eigenvalues")
				plt.plot(eigens, ".")
				plt.subplot(122)
				plt.title("Eigenvalue Spacing Distribution")
				plt.hist(compute_spacings(eigens))
				plt.savefig("rmt_%03d.png" % (int(threshold * 1000)))
				plt.close()

			# save most recent chi-square value less than critical value
			if chi < 99.607:
				final_chi = chi
				final_threshold = threshold

			# save largest chi-square value which occurs after final_chi
			if final_chi < 99.607 and chi > final_chi:
				max_chi = chi

		# output results of threshold test
		print("%f\t%d\t%f" % (threshold, S_pruned.shape[0], chi))

		# decrement threshold and fail if minimum threshold is reached
		threshold -= args.TSTEP
		if threshold < args.TSTOP:
			print("error: could not find an adequate threshold above stopping threshold")
			sys.exit(0)

	return final_threshold



if __name__ == "__main__":
	# define threshold methods
	METHODS = {
		"powerlaw": powerlaw,
		"rmt": rmt
	}

	# parse command-line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", required=True, help="correlation matrix file", dest="INPUT")
	parser.add_argument("--genes", type=int, required=True, help="number of genes", dest="NUM_GENES")
	parser.add_argument("--method", default="rmt", choices=["powerlaw", "rmt"], help="thresholding method", dest="METHOD")
	parser.add_argument("--tstart", type=float, default=0.99, help="starting threshold", dest="TSTART")
	parser.add_argument("--tstep", type=float, default=0.001, help="threshold step size", dest="TSTEP")
	parser.add_argument("--tstop", type=float, default=0.5, help="stopping threshold", dest="TSTOP")
	parser.add_argument("--spline", action="store_true", help="whether to use spline interpolation", dest="SPLINE")
	parser.add_argument("--minclus", type=int, default=1, help="minimum clusters", dest="MIN_CLUSTERS")
	parser.add_argument("--maxclus", type=int, default=5, help="maximum clusters", dest="MAX_CLUSTERS")
	parser.add_argument("--visualize", action="store_true", help="whether to visualize results", dest="VISUALIZE")

	args = parser.parse_args()

	# print arguments
	pprint.pprint(vars(args))

	# load data
	cmx = pd.read_csv(args.INPUT, sep="\t")

	# initialize method
	compute_threshold = METHODS[args.METHOD]

	# compute threshold
	threshold = compute_threshold(args)

	print("%0.3f" % (threshold))
