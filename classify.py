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

def analyze(age, gender, pd_status, lipid_names, lipids, out, num_iter):
  # clone pd_status
  N = len(lipid_names)
  pd_status_cp = np.empty_like (pd_status)
  pd_status_cp[:] = pd_status
  min_p_scores = []
  ttest = {}
  stat_summary = {}
  # calculates p-values after shuffling the PD status.
  # repeat 1000 times.
  for k in range(0, num_iter + 1):
    mean = []
    residual = []
    min_p_score = 1000
    for i in range(0, N):
      # fit regression line and get residuals
      response = [row[i] for row in lipids]
      x = [[gender[j], age[j]] for j in range(0, len(gender))]
      #x = [age[j] for j in range(0, len(age))] # if stratifying gender
      sm.add_constant(age)
      est = sm.OLS(response, x)
      est = est.fit()
      # regression coefficients
      #stat_summary[lipid_names[i]] = \
      #    (est.summary().tables[1].data[1][1], est.summary().tables[1].data[1][2])
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
    # record min_p_score
    if k > 0:
      min_p_scores.append(min_p_score)
    if k % 50 == 0:
      print k
    np.random.shuffle(pd_status_cp)

    #return stat_summary

  for i in range(0, N):
    old_t, old_p = ttest[lipid_names[i]]
    adjusted_p = len(filter(lambda x: x < old_p, min_p_scores)) / float(num_iter)
    ttest[lipid_names[i]] = (old_t, old_p, adjusted_p)

  with open(out, 'a') as f:
    f.write(json.dumps(ttest))

  # dendrogram for hierarchical clustering
  #tree = hcluster.hcluster(residual)
  #hcluster.drawdendrogram(tree, ['figures/' + str(i) + '.png' for i in range(0, N)], jpeg='cluster.jpg')

def analyze_lipids():
  # prepare/cleanse data
  df = pd.read_csv('plasma_all_values_transposed.csv')[:250]
  df.drop(df.columns[[53]], axis=1, inplace=True) # drop empty column
  cols = list(df.columns.values)
  age = df['Age'].astype(int)
  gender = df['Gender'].apply(lambda x: 0 if x=='M' else 1).astype(float)
  pd_status = df['PD_status'].apply(lambda x: 1 if x=='PD' else 0).astype(bool)
  # columns 12:252X 12:764
  lipid_names = np.array(df.columns.values[15:766])
  #print lipid_names
  lipids = df.ix[:250,15:766].astype(float)
  # remove rows with empty data
  valid_rows = np.all(np.isfinite(lipids), axis=1)
  lipids = lipids[valid_rows]
  gender = np.array(gender[valid_rows])
  nonzero_cols = np.where(lipids.any(axis=0))[0]
  lipid_names = lipid_names[nonzero_cols]
  lipids = lipids[nonzero_cols]
  N = len(lipid_names)
  age = np.array(age[valid_rows])
  pd_status = np.array(pd_status[valid_rows])

  '''
  # block for stratifying male
  filter_male = list(gender.apply(lambda x: False if x==1 else True))
  #print filter_male
  lipids = lipids[filter_male]
  gender = np.array(gender[filter_male])
  age = np.array(age[filter_male])
  pd_status = np.array(pd_status[filter_male])
  '''
  lipids = np.ma.array(lipids, mask=False)
  quantile_norm(lipids)
  #np.savetxt('quantile_norm_lipids.csv', lipids, delimiter=",")
  analyze(age, gender, pd_status, lipid_names, lipids, 'ttest_10000.json', 10000)

def analyze_links():
  df = pd.read_csv('link_values.csv')[:246]
  age = df['Age'].astype(int)
  gender = df['Gender'].astype(int)
  pd_status = df['PD_status'].astype(bool)
  lipid_names = np.array(df.columns.values[3:130])
  lipids = df.ix[:246,3:130].astype(float)


  '''
  # block for stratifying female
  filter_female = list(gender.apply(lambda x: False if x==0 else True))
  print filter_female
  lipids = lipids[filter_female]
  gender = np.array(gender[filter_female])
  age = np.array(age[filter_female])
  pd_status = np.array(pd_status[filter_female])
  '''

  lipids = np.ma.array(lipids, mask=False)
  quantile_norm(lipids)
  analyze(np.array(age), np.array(gender), np.array(pd_status), lipid_names, lipids, 'ttest_links.json', 1000)

def get_link_values():
  df = pd.read_csv('plasma_all_values_transposed.csv')[:250]
  df.drop(df.columns[[53]], axis=1, inplace=True) # drop empty column
  age = df['Age'].astype(int)
  gender = df['Gender'].apply(lambda x: 0 if x=='M' else 1).astype(float)
  pd_status = df['PD_status'].apply(lambda x: 1 if x=='PD' else 0).astype(bool)
  lipid_names = np.array(df.columns.values[15:766])
  #print lipid_names
  lipids = df.ix[:250,15:766].astype(float)
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
  lipids = np.ma.array(lipids, mask=False)
  quantile_norm(lipids)

  trans = pd.read_csv('translation.csv')[:21]
  short = trans['Short name']
  to = trans['To']
  dic = {}
  for i in xrange(len(short)):
    dic[short[i]] = to[i].split(';')
  print dic
  link_lipid_names = []
  filter_index = []
  for i, name in enumerate(lipid_names):
    if name.split()[0] in list(short):
      link_lipid_names.append(name)
      filter_index.append(i)

  print link_lipid_names
  print filter_index
  lipids = [row[filter_index] for row in lipids]
  print len(link_lipid_names), len(lipids[0])

  columns = []
  first = True
  link_values = []
  for row in lipids:
    new_row = []
    for source_i, name in enumerate(link_lipid_names):
      source, rest = name.split() if len(name.split()) == 2 else (name.split()[0], '')
      for target in dic[source]:
        target_name = target + ' ' + rest if rest else target
        if target_name in link_lipid_names:
          target_i = link_lipid_names.index(target_name)
        else:
          #print "Couldn't find " + target_name + " from " + name
          continue
        if target + '-' + name not in columns:
          if first:
            columns.append(source + '-' + target_name)
          new_row.append(abs(row[source_i] - row[target_i]))
    if first:
      first = False
    link_values.append(new_row)

  print len(age), len(gender), len(pd_status), len(lipids)
  with open('link_values.csv', 'a') as f:
    f.write(','.join(['Age', 'Gender', 'PD_status'] + columns) + '\n')
    for i,row in enumerate(link_values):
      f.write(str(age[i]) + ',' + str(gender[i]) + ',' + str(pd_status[i]) + ',' + ','.join(str(value) for value in row) + '\n')

def json_to_csv(fn):
  d = json.load(open(fn, 'r'))
  lines = ''
  for lipid in d:
    lines += lipid + ',' + ','.join(map(str, d[lipid])) + '\n'
  print lines

if __name__=="__main__":
  parser = optparse.OptionParser()
  parser.add_option('-p', '--plot', dest='plot', action="store_true", default=False)
  parser.add_option('-c', '--lipids', dest='lipids', action="store_true", default=False)
  parser.add_option('-l', '--link', dest='link', action="store_true", default=False)
  opts, args = parser.parse_args()
  if opts.plot:
    plot()
  elif opts.lipids:
    analyze_lipids()
  elif opts.link:
    analyze_links()
  else:
    main()
