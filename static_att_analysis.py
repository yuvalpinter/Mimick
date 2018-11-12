from __future__ import division
import sys
import argparse
from collections import Counter
from copy import deepcopy
from numpy import random

from utils import split_tagstring
from evaluate_morphotags import Evaluator
from prop_weigh import sqrt_w, log_w, inv_w, linear_w, sum_w, sq_w, rand_w

def read_results_file(in_file, attval):
    """
    :return: list of {gold tags, observed tags} tuples, and an Evaluator object
    """
    golds = []
    obsvs = []
    mode = 'att_val' if attval else 'att'
    ev = Evaluator(m=mode)
    gold_insts = 0
    errs = Counter()
    with open(in_file) as in_lines:
        for l in in_lines.readlines():
            cols = l.split('\t')
            if len(cols) < 3: continue
            g = split_tagstring(cols[2])
            o = split_tagstring(cols[4])
            golds.append(g)
            obsvs.append(o)
            ev.add_instance(g, o)
            gold_insts += len(g)
            for att,val in g.items():
                if att not in o:
                    errs[(att, val, None)] += 1
                elif val != o[att]:
                    errs[(att, val, o[att])] += 1
            for att,val in o.items():
                if att not in g:
                    errs[(att, None, val)] += 1
    return (golds, obsvs), ev, gold_insts, errs

def err_fix(err_type, ev):
    att, gt, ot = err_type
    assert gt != ot
    if ev.mode == 'att':
        if gt == None:
            # false positive
            ev.observed[att] -= 1
        elif ot == None:
            # false negative
            ev.observed[att] += 1
            ev.correct[att] += 1
        else:
            # misclassification
            ev.correct[att] += 1
    elif ev.mode == 'att_val':
        if gt == None:
            # false positive
            ev.observed[(att, ot)] -= 1
        elif ot == None:
            # false negative
            ev.observed[(att, gt)] += 1
            ev.correct[(att, gt)] += 1
        else:
            # misclassification
            ev.observed[(att, ot)] -= 1
            ev.observed[(att, gt)] += 1
            ev.correct[(att, gt)] += 1
    else:
        raise Exception('Evaluation mode not supported')

def agg_inc_check(errs, ev, weigh):
    """
    incrementally checks for improvements, based on error type counts
    changes both inputs in-place
    :return: updated error type
    """
    max_score = ev.mac_f1(weigh=wg)
    argmax_score = None
    ties = 0
    for tp,c in errs.items():
        if c <= 0:
            # already fixed all errors of this type
            continue
        
        # calculate score and replace max, argmax, if winner
        ev2 = deepcopy(ev)
        err_fix(tp, ev2)
        score = ev2.mac_f1(weigh=wg)
        if score > max_score:
            max_score = score
            argmax_score = tp
            ties = 0
        elif score == max_score:
            ties += 1
            # reservoir sample new correction type
            if random.randint(ties+1) == 0:
                argmax_score = tp            
        
    # update ev and errs with argmax
    if argmax_score is not None:
        err_fix(argmax_score, ev)
        errs[argmax_score] -= 1
    
    return argmax_score, ties
    

def thorough_inc_check(exs, ev):
    """
    checks incrementally for improvements, one test instance at a time
    :return: index of improvement, attribute changed, new value
    """
    gs, obss = exs
    # TODO...

parser = argparse.ArgumentParser()
parser.add_argument("--infile")
parser.add_argument("--outfile", default=None)
parser.add_argument("--attval", action='store_true', help="use macro of <attribute, value> pairs")
parser.add_argument("--fixes", type=int, default=10, help="number of fixing iterations to run, default - 10")
parser.add_argument("--weighting", default='sum', help="weighting scheme. supported - log, sqrt, sum (macro, default), linear (micro), inv, sq, rand")
parser.add_argument("--v", type=int, default=0, help="verbosity, range 0-3 (default 0)")
options = parser.parse_args()

if options.v > 0:
    print 'reading from file', options.infile
    if options.attval:
        print 'using attval mode'
exs, ev, gold_insts, errs = read_results_file(options.infile, options.attval)
if options.v > 0:
    print 'read', len(exs[0]), 'instances with', gold_insts, 'tags and', sum(errs.values()), 'prediction errors across', len(errs), 'error types'
    if options.v > 2:
        print errs.most_common()

if options.v > 0:
    print 'calibration metric', options.weighting
    
if options.weighting == 'log':
    wg = log_w
elif options.weighting == 'sqrt':
    wg = sqrt_w
elif options.weighting == 'inv':
    wg = inv_w
elif options.weighting == 'sq':
    wg = sq_w
elif options.weighting == 'rand':
    wg = rand_w
elif options.weighting == 'linear':
    wg = linear_w
else:
    wg = sum_w

print '==========================='
print 'basic macro F1:', ev.mac_f1()
print 'basic weighted macro F1:', ev.mac_f1(weigh=wg)
print 'basic micro F1:', ev.mic_f1()

error_improvements = []
tieses = []
fixes_left = min(options.fixes, sum(errs.values()))
if options.v > 0:
    print 'performing', fixes_left, 'fixes'
    if options.outfile is not None:
        print 'writing decisions to file', options.outfile
        out_writer = open(options.outfile, 'w')
        out_writer.write('att\tgold\tobserved\tnew f1\tties\n')
while fixes_left >= 0:
    impr, ties = agg_inc_check(errs, ev, weigh=wg)
    error_improvements.append(impr)
    tieses.append(ties)
    if options.v > 1:
        att, g, o = impr
        print '{}\t{}\t{}\t{:.5f}\t{}'.format(att, g, o, ev.mac_f1(weigh=wg), ties)
    if options.outfile is not None:
        att, g, o = impr
        out_writer.write('{}\t{}\t{}\t{:.5f}\t{}\n'.format(att, g, o, ev.mac_f1(weigh=wg), ties))
    fixes_left -= 1

print '==========================='
print 'final macro F1:', ev.mac_f1()
print 'final weighted macro F1:', ev.mac_f1(weigh=wg)
print 'final micro F1:', ev.mic_f1()
if options.v > 0:
    print 'total number of ties:', sum(tieses)
print '==========================='
