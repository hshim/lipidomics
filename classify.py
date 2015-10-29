#!/usr/bin/python
from __future__ import division
import pprint
import json
from StringIO import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np
import numpy.ma as ma
from numpy import genfromtxt, savetxt
from scipy import interpolate, stats
import random
import optparse
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import hcluster

pp = pprint.PrettyPrinter(indent=2)
def loadData():
  dataset = genfromtxt(open('clinical_vars.csv','r'), delimiter=',', dtype=None, skip_header=1)
  dataset = filter(lambda r: r[6] == 'White', dataset)
  return dataset

# copied from https://github.com/andrewdyates/quantile_normalize
def my_interpolate(pos, v):
  """Return interpolated value pos from v.
  Args:
    pos: 0 <= x <= 1 fractional position in v
    v: [num] vector
  Returns:
    num of interpolated v @ pos
  """
  n = len(v)-1
  low, high = int(np.floor(n*pos)), int(np.ceil(n*pos))
  if low==high:
    return v[low]
  else:
    frac = pos*n - low
    return v[low]*(1-frac) + v[high]*(frac)


def frac_intervals(n):
  """n intervals uniformly spaced from 0 to 1 inclusive"""
  q = np.arange(0,n)/(n-1)
  q[0],q[-1] = 0, 1
  return q

def quantile_norm(M):
  """Quantile normalize masked array M in place."""
  Q = M.argsort(0, fill_value=np.inf)
  m, n = np.size(M,0), np.size(M,1)
  # np.count_nonzero changed to np.sum for numpy1.5
  counts = np.array([m - np.sum(M.mask[:,i]) for i in range(n)])

  # compute quantile vector
  quantiles = np.zeros(m)
  for i in xrange(n):
    # select first [# values] rows of argsorted column in Q
    r = counts[i] # number of non-missing values for this column
    v = M.data[:,i][Q[:r,i]] # ranks > r point to missing values == infinity
    # create linear interpolator for existing values
    f = interpolate.interp1d(np.arange(r)/(r-1), v)
    v_full = f(frac_intervals(m))

    quantiles += v_full
  quantiles = quantiles / n
  f_quantile = interpolate.interp1d(frac_intervals(m), quantiles)

  ranks = np.empty(m, int)
  for i in xrange(n):
    r = counts[i]
    ranks[Q[:,i]] = np.arange(m)
    # Get equivalence classes; unique values == 0
    dupes = np.zeros(m, dtype=np.int)
    for j in xrange(r-1):
      if M[Q[j,i],i] == M[Q[j+1,i],i]:
        dupes[j+1] = dupes[j]+1
    # zero-out ranks higher than the number of values (to prevent out of range errors)
    ranks[ranks>=r] = 0
    # Replace column with quantile ranks
    M.data[:,i] = f_quantile(ranks/(r-1))
    # Average together equivalence classes
    j = r-1
    while j >= 0:
      if dupes[j] == 0:
        j -= 1
      else:
        idxs = Q[j-dupes[j]:j+1,i]
        M.data[idxs,i] = np.median(M.data[idxs,i])
        j -= 1 + dupes[j]
    assert j == -1

def plot():
  #df = pd.read_csv('clinical_vars.csv')
  df = pd.read_csv('plasma_data_transpose.csv')
  #df = df[(df['DEM_7_ETHNICITY'] == 'White') & (df['MED_34b_STAT (taking drugs treating lipid levels?)'] == 'No')]
  cols = list(df.columns.values)
  #severities = cols[17:22]
  #enzymes = cols[25:]
  enzymes = cols[12:51]
  #normalized = [enzymes[i] for i in range(0, len(enzymes), 2)]
  #num_enzymes = len(normalized)
  num_enzymes = len(enzymes)
  print enzymes

  name = 0
  for i in range(num_enzymes):
    for j in range(i+1, num_enzymes):
      enzyme1 = enzymes[i]
      enzyme2 = enzymes[j]

      df.head()

      # setup figure
      plt.figure(figsize=(10, 8))
      plt.title("No medication")

      # scatter plot of balance (x) and income (y)
      ax1 = plt.subplot(221)
      cm_bright = ListedColormap(['#FF0000', '#0000FF'])
      ax1.scatter(df[enzyme1], df[enzyme2], c=(df['PD_status'] == 'PD'), cmap=cm_bright)
      ax1.set_xlim((df[enzyme1].min(), df[enzyme1].max()))
      ax1.set_ylim((df[enzyme2].min(), df[enzyme2].max()))
      ax1.set_xlabel(enzyme1)
      ax1.set_ylabel(enzyme2)
      ax1.legend(loc='upper right')
      plt.savefig(str(name) + '.png')
      name += 1

def main():
  dataset = loadData()
  test_samples = []
  train = []
  target = []
  for r in dataset:
    sample = [r[i] for i in range(25, len(r), 2)]
    if random.randint(0,10) == 1:
      test_samples.append((sample, r[1] == 'PD'))
    else:
      train.append(sample)
      target.append(r[1] == 'PD')
  rf = RandomForestClassifier(n_estimators=100)
  rf.fit(train, target)
  if len(test_samples) > 0:
    res = rf.predict([test_sample[0] for test_sample in test_samples])
    res = [res[i] == test_samples[i][1] for i in range(0, len(test_samples))]
    print res
    print "Got " + str(sum(res)) + " out of " + str(len(res))

def cluster():
  # prepare/cleanse data
  df = pd.read_csv('plasma_data_transpose.csv')[:250]
  cols = list(df.columns.values)
  age = df['Age'].astype(float)
  gender = df['Gender'].apply(lambda x: 0 if x=='M' else 1).astype(float)
  pd_status = df['PD_status'].apply(lambda x: 1 if x=='PD' else 0).astype(bool)
  # columns 12:252
  lipid_names = np.array(df.columns.values[12:252])
  lipids = df.ix[:250,12:252].astype(float)
  # remove rows with empty data
  valid_rows = np.all(np.isfinite(lipids), axis=1)
  lipids = lipids[valid_rows]
  gender = np.array(gender[valid_rows])
  nonzero_cols = np.where(lipids.any(axis=0))[0]
  lipid_names = lipid_names[nonzero_cols]
  lipids = lipids[nonzero_cols]
  N = len(lipid_names)
  print N
  age = np.array(age[valid_rows])
  pd_status = np.array(pd_status[valid_rows])
  #print pd_status
  #print age
  #print gender
  #print lipid

  lipids = np.ma.array(lipids, mask=False)
  quantile_norm(lipids)
  #np.savetxt('quantile_norm_lipids.csv', lipids, delimiter=",")

  # clone pd_status
  pd_status_cp = np.empty_like (pd_status)
  pd_status_cp[:] = pd_status
  min_p_scores = []
  ttest = {}
  # calculates p-values after shuffling the PD status.
  # repeat 1000 times.
  for k in range(0, 1001):
    mean = []
    residual = []
    min_p_score = 1000
    for i in range(0, N):
      # fit regression line and get residuals
      response = [row[i] for row in lipids]
      x = [[gender[j], age[j]] for j in range(0, len(gender))]
      sm.add_constant(age)
      est = sm.OLS(response, x)
      est = est.fit()
      ys = [est.predict(pt) for pt in x]
      current_mean = np.mean(ys)
      mean.append(current_mean)
      residual_i = [y - current_mean for y in response]

      # get t-test score and p-value
      pd_group = []
      ctrl_group = []
      j = 0
      for is_pd in pd_status_cp:
        if is_pd:
          pd_group.append(residual_i[j])
        else:
          ctrl_group.append(residual_i[j])
        j += 1

      t, p = stats.ttest_ind(pd_group, ctrl_group)
      # update min_p_score for this k
      if p < min_p_score:
        min_p_score = p
      # record t, p values for unshuffled data
      if k == 0:
        residual.append(residual_i)
        ttest[lipid_names[i]] = (t.item(), p)
        if lipid_names[i] in ['SM d18:1/20:1', 'GM3', 'SM d18:1/22:1']:
          print lipid_names[i], residual_i, t, p
    # record min_p_score
    if k > 0:
      min_p_scores.append(min_p_score)
    print k
    np.random.shuffle(pd_status_cp)

  for i in range(0, N):
    old_t, old_p = ttest[lipid_names[i]]
    adjusted_p = len(filter(lambda x: x < old_p, min_p_scores)) / 1000.0
    ttest[lipid_names[i]] = (old_t, old_p, adjusted_p)
  print json.dumps(ttest)

  #tree = hcluster.hcluster(residual)

  #hcluster.drawdendrogram(tree, [str(i) + '.png' for i in range(0, N)], jpeg='cluster.jpg')
  #return tree

def json_to_csv(fn):
  d = json.load(open(fn, 'r'))
  lines = ''
  for lipid in d:
    lines += lipid + ',' + ','.join(map(str, d[lipid])) + '\n'
  print lines

if __name__=="__main__":
  parser = optparse.OptionParser()
  parser.add_option('-p', '--plot', dest='plot', action="store_true", default=False)
  parser.add_option('-c', '--cluster', dest='cluster', action="store_true", default=False)
  opts, args = parser.parse_args()
  if opts.plot:
    plot()
  elif opts.cluster:
    cluster()
  else:
    main()
