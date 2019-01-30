from random import randint
from math import sqrt
from itertools import product
import os.path

import matplotlib.pyplot as plt; plt.style.use('ggplot')
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from sklearn.metrics import euclidean_distances
from pyemd import emd, emd_with_flow
import gensim
# from MulticoreTSNE import MulticoreTSNE as TSNE


class DirtPile():
    def __init__(self, position, mass, label=None):
        self.position = position
        self.x, self.y = position
        self.mass = mass
        if not label:
            self.label = "{mass} @ ({x}, {y})".format(mass=self.mass, x=self.x, y=self.y)
        else:
            self.label = label
            
    def __str__(self):
        # return "{mass} @ ({x}, {y})".format(mass=self.mass, x=self.x, y=self.y)
        return self.label + " "


class PileDistribution():
    def __init__(self, *piles):
        self.piles = list(piles)
        self.masses = {tuple(p.position): p.mass for p in self.piles}
        self.mass_sum = sum(p.mass for p in self.piles)
        
    def __str__(self):
        # return '\n'.join([str(pile) for pile in self.piles])
        return ''.join([str(pile) for pile in self.piles])
    
    def __getitem__(self, index):
        return self.piles[index]

def plot_dirt_piles(pd1, pd2, normed=True, r_scale=5000, figsize=(8,8), annotate=True):
    p1_x = [pile.x for pile in pd1.piles]
    p1_y = [pile.y for pile in pd1.piles]

    p2_x = [pile.x for pile in pd2.piles]
    p2_y = [pile.y for pile in pd2.piles]
    
    if normed:
        p1_masses = [pile.mass / pd1.mass_sum for pile in pd1.piles]
        p2_masses = [pile.mass / pd2.mass_sum for pile in pd2.piles]
    else:
        p2_masses = [pile.mass for pile in pd2.piles]
        p1_masses = [pile.mass for pile in pd1.piles]


    plt.figure(figsize=figsize)
    plt.scatter(x=p1_x, y=p1_y, s=[r_scale*m for m in p1_masses], c='r', alpha=0.5)
    plt.scatter(x=p2_x, y=p2_y, s=[r_scale*m for m in p2_masses], c='b', alpha=0.5)

    if annotate:
        for pile in pd1.piles:
            plt.annotate(xy=[pile.x, pile.y], s=pile.label, textcoords='data')
        for pile in pd2.piles:
            plt.annotate(xy=[pile.x, pile.y], s=pile.label, textcoords='data')

    # plt.show()

def generate_signatures(piledist1, piledist2, normalize=False):
    
    # build unique list of pile positions
    # sorted by distance from the origin
    all_piles = piledist1.piles + piledist2.piles
    positions = sorted(list(set(pile.position for pile in all_piles)),
                       key=lambda x: sqrt(x[0]**2 + x[1]**2))
    
    # build signatures
    # check if the distribution has a mass at this position or return 0
    p1_signature = []
    p2_signature = []
    for position in positions:
        p1_location_mass = piledist1.masses.get(position, 0)
        p2_location_mass = piledist2.masses.get(position, 0)
        p1_signature.append(p1_location_mass)
        p2_signature.append(p2_location_mass)
    if normalize:
        p1_signature = [mass / sum(p1_signature) for mass in p1_signature]
        p2_signature = [mass / sum(p2_signature) for mass in p2_signature]
    
    return positions, p1_signature, p2_signature


def calculate_emd(signature_1, signature_2, distance_matrix):
    first_signature = np.array(signature_1, dtype=np.double)
    second_signature = np.array(signature_2, dtype=np.double)
    distances = np.array(distance_matrix, dtype=np.double)
    emd, flow = emd_with_flow(first_signature, second_signature, distances)
    flow = np.array(flow)
    return emd, flow


def plot_emd_solution(pd1, pd2, positions, emd, flow, normed=True, r_scale=5000, figsize=(8,8), annotate=True):
    p1_x = [pile.x for pile in pd1.piles]
    p1_y = [pile.y for pile in pd1.piles]

    p2_x = [pile.x for pile in pd2.piles]
    p2_y = [pile.y for pile in pd2.piles]
    
    if normed:
        p1_masses = [pile.mass / pd1.mass_sum for pile in pd1.piles]
        p2_masses = [pile.mass / pd2.mass_sum for pile in pd2.piles]
    else:
        p2_masses = [pile.mass for pile in pd2.piles]
        p1_masses = [pile.mass for pile in pd1.piles]
        
    flow_measures = []
    for to_pos_ix, from_pos_ix in zip(*np.nonzero(flow)):
        to_pos = positions[to_pos_ix]
        from_pos = positions[from_pos_ix]
        measure = {'to' : to_pos, 'from' : from_pos,
                   'xs' : (to_pos[0], from_pos[0]),
                   'ys' : (to_pos[1], from_pos[1]),
                   'value' : flow[to_pos_ix, from_pos_ix]}
        flow_measures.append(measure)

    plt.figure(figsize=figsize)

    plt.scatter(x=p1_x, y=p1_y, s=[r_scale*m for m in p1_masses], c='r', alpha=0.8)   
    plt.scatter(x=p2_x, y=p2_y, s=[r_scale*m for m in p2_masses], c='b', alpha=0.8)

    for measure in flow_measures:
        plt.plot([*measure['xs']], [*measure['ys']],
                 color='black', lw=measure['value']*r_scale/100, alpha=0.7, solid_capstyle='round')

    ("Example Earth Movers Distance Solution\n EMD: {0:.2f}".format(emd))

    plt.show()
