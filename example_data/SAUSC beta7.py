# -*- coding: utf-8 -*-
"""
Written by Josh Bishop and Luke Smith

University of Oxford

07/03/2021
"""

from matplotlib.colors import LinearSegmentedColormap
import numpy as np


try:
    from scipy import stats
except ImportError:
    raise Exception("Please install scipy via 'pip install scipy'.")

try:
    import matplotlib.cm as cm
except ImportError:
    raise Exception("Please install matplotlib via 'pip install matplotlib'.")

import matplotlib.pyplot as plt
import copy
from typing import List, NamedTuple, Dict
from tkinter import filedialog
import tkinter as tk


if __name__ == 'pymol':
    from pymol import cmd

# Needed for pymol porting
def _check_str_bool(string):
    if string.lower() == 'true': return True
    elif string.lower() == 'false': return False
    else: return TypeError('input not converted to bool, write true or false')

def HDX_excel_to_dict(
        filepath: str
        ):
    '''
    Converts an excel into a dictionary
    '''
    
    assert 'z' not in np.loadtxt(filepath, delimiter=',', dtype=object)[0], 'File is cluster data not state data!'
        
    data = np.loadtxt(filepath, delimiter=',', dtype=object, usecols=(list(range(16))))
    headers = data[0]
    data = data[1:].T
    results = {str(header).lower().replace(' ', '_'): arr for header, arr in zip(headers, data)}
    
    for key, array in results.items():
        try:
            results[key] = array.astype(float)
        except ValueError:
            pass
    
    return results

class HDXParameters:
    '''
    Accessory class to interpret HDX results
    '''
    def __init__(
            self, 
            states: set, 
            exposures: set
            ):
        self.states = states
        self.exposures = exposures
        self.block_size = len(states) * len(exposures)
    
    def __repr__(self):
        return self.__dict__.__repr__()

class HDXResults:
    '''
    Holder class for all results, allows smart indexing and custom parsing
    '''
    def __init__(self, dict):
        self.__dict__.update(dict)
        if not issubclass(type(self.exposure), float):
            self.parameters = HDXParameters(
                states = set(self.state),
                exposures = set(self.exposure)
                )
        
    def __getitem__(self, index):
        return HDXResults({attr:values[index] for attr, values in self.__dict__.items() if attr != 'parameters'})
    
    def __len__(self):
        return len(self.start)
    
    def parse(self, length):
        indexes = np.reshape(np.arange(len(self)), (-1, length))
        for i in indexes:
            yield self[i]


class Residue:
    
    def __init__(
            self,
            n: int,
            mean: float,
            std: float
            ):
        
        self.n = n
        self.mean = mean
        self.std = std
        self.colour = None

class StatsResult:
    
    def __init__(
            self,
            uptake_difference: float,
            relative_uptake_difference: float,
            p_value: float,
            significant: bool,
            raw_colour: tuple = None,
            rel_colour: tuple = None,
            ):
        
        self.uptake_difference = uptake_difference
        self.relative_uptake_difference = relative_uptake_difference
        self.p_value = p_value
        self.significant = significant
        self.raw_colour = raw_colour
        self.rel_colour = rel_colour
        self.neglogp = -np.log10(self.p_value)
    
    def __repr__(self):
        return str(self.__dict__)

class StatsResultsWrapper:
    
    def __init__(self, results):
        self.__dict__.update(
                {key: np.array([r.__dict__[key] for r in results])
                for key in results[0].__dict__.keys()}
                )
        
        
    
def residue_averaged_exposure(
        sequences, 
        data: StatsResultsWrapper,
        significant_only: bool = False
        )->List[Residue]:
    
    '''
    Return a list of averaged values per residue, and residues involved in cases of no coverage
    '''
    
    # Get residue data
    max_residue = np.nanmax([s.end for s in sequences])
    residues = np.arange(0, max_residue+1)
    
    mean, std = [], []
    
    for r in residues:
        residue_in_seq = np.array([s.residue_present(r) for s in sequences])
        
        # No coverage
        if not np.any(residue_in_seq):
            mean.append(np.nan)
            std.append(np.nan)
            continue
        
        sel = (residue_in_seq & data.significant) if significant_only else residue_in_seq
        
        # None significant
        if not np.any(sel):
            mean.append(0)
            std.append(0)
            continue
        
        else:
            mean.append(np.nanmean(data.uptake_difference[sel]))
            std.append(np.nanstd(data.uptake_difference[sel]))
    
    data = [Residue(n=n, mean=mean, std=std) for n, mean, std in zip(residues, mean, std)]
    
    return data
    
class QuickSeq:
    
    def __init__(self, start, end, contributions=None):
        self.start = start
        self.end = end
        self.res = set(range(start, end+1))
        self.contributions = contributions
    
    def __hash__(self):
        return hash(tuple([self.start, self.end]))
    
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
    
    def __repr__(self):
        return f'{self.start} - {self.end}'
    
def calculate_intersections(sequences):
    max_iter = 100
    solved = False
    
    new_sequences = {QuickSeq(s.start, s.end, {s}) for s in sequences}
    
    for counter in range(max_iter):
        print(f'Round {counter}')
        test_sequences = {i for i in new_sequences}
        intersections = {i : {j : i.res^j.res for j in new_sequences if (len(i.res^j.res) > 0) and (len(i.res^j.res) < len(i.res))} for i in new_sequences}
        for seq1, intersection in intersections.items():
            for seq2, overlap in intersection.items():
                if sorted(overlap) == list(range(min(overlap), np.nanmax(overlap)+1)):
                    cont = set()                    
                    for i in seq1.contributions:
                        cont.add(i)
                    for i in seq2.contributions:
                        cont.add(i)
                        
                    new_seq = QuickSeq(
                        start = min(overlap),
                        end = np.nanmax(overlap),
                        contributions = cont
                        )
                    
                    new_sequences.add(new_seq)
                    
                    
        solved = test_sequences == new_sequences
        if solved:
            print(f'Solution found in {counter} iterations')
            break
        if counter == max_iter -1:
            print('Max iteration reached')
    
    
    
    
#############################


pooled_sd = lambda sd: np.sqrt(np.sum(np.array(sd)[np.array(sd) != 0] ** 2) / len(sd))

def pooled_standard_error_mean(
        standard_deviations: np.ndarray,
        num_repeats: int
        ) -> float:
    '''
    Standard deviations represents multiple time points, each having the same sample size.
    '''
    
    pooled_sem = np.sqrt(
        (2 * (pooled_sd(standard_deviations)**2)) / ((num_repeats * 2) - 2)
        )
    
    return pooled_sem

######### PLOTTING ###########
    
# needs Luke!

def _cumulative_volcano_plot(
        results: StatsResultsWrapper,
        sem: float,
        confidence_interval: float,
        annotations: List[str] = None,
        uptake: str = 'abs',
        colour: str = 'rel',
        xpadding = 1.05,
        insignificant_colour: tuple = (0.7, 0.7, 0.7),
        size = 20,
        fontsize: int = 15,
        ):
    
    valid = ['rel', 'abs']
    assert (uptake in valid) and (colour in valid),  "must select either 'abs' (actual uptake difference) or 'rel' (relative uptake difference)"
    
    # Extract data
    x = results.uptake_difference if uptake == 'abs' else results.relative_uptake_difference 
    y = results.neglogp
    colours = results.raw_colour if colour == 'abs' else results.rel_colour
    colours[~results.significant] = insignificant_colour
    
    
    fig, ax = plt.subplots(figsize=(5,5))
    xlabel = 'Uptake difference (Da)' if uptake == 'abs' else 'Relative uptake difference (%)'
    ax.set(
        ylabel = r'$-log_{10}(P)$', 
        xlim = (-np.nanmax(abs(x)) * xpadding, np.nanmax(abs(x)) * xpadding), 
        xlabel = xlabel
        )
    
    ax.scatter(x, y, c=colours, s=size)
    
    # Lines to show statistical limits
    ax.axhline(-np.log10(1-confidence_interval), color='black', linestyle = '--', zorder=-1)
    if uptake == 'abs':
        ax.axvline(sem, color='black', linestyle = '--', zorder=-1)
        ax.axvline(-sem, color='black', linestyle = '--', zorder=-1)
    
    if annotations:
        for x, y, string in zip(x[results.significant], y[results.significant], np.array(annotations)[results.significant]):
            ax.annotate(string, (x,y), fontsize=fontsize)
    
    plt.suptitle('Cumulative volcano plot')
    plt.show()
    
    
    
    
def _volcano_plot(
        all_results: dict,
        sem: float,
        confidence_interval: float,
        annotations: List[str] = None,
        uptake: str = 'abs',
        colour: str = 'rel',
        xpadding = 1.05,
        insignificant_colour: tuple = (0.7, 0.7, 0.7),
        size = 20,
        fontsize: int = 5,
        ):
    
    valid = ['rel', 'abs']
    assert (uptake in valid) and (colour in valid),  "must select either 'abs' (actual uptake difference) or 'rel' (relative uptake difference)"
    
    xmax = np.nanmax([np.nanmax(abs(r.uptake_difference)) for r in all_results.values()]) if uptake=='abs' else np.nanmax([np.nanmax(abs(r.relative_uptake_difference)) for r in all_results.values()])
    ymax = np.nanmax([np.nanmax(abs(r.neglogp)) for r in all_results.values()])
    
    
    fig, axes = plt.subplots(1, len(all_results), sharey=True, sharex=True, figsize=(3*len(all_results), 3))
    xlabel = 'Uptake difference (Da)' if uptake == 'abs' else 'Relative uptake difference (%)'
    axes[0].set(
            ylabel = r'$-log_{10}(P)$', 
            xlim = (-xmax * xpadding, xmax * xpadding),
            ylim = (0, ymax)
            )
    
    
    for ax, (time_label, results) in zip(axes, all_results.items()):
        
        ax.set(xlabel = xlabel, title=f'{round(time_label, 2)} min')
        x = results.uptake_difference if uptake == 'abs' else results.relative_uptake_difference
        y = results.neglogp
        colours = results.raw_colour if colour == 'abs' else results.rel_colour
        colours[~results.significant] = insignificant_colour
        
        scat = ax.scatter(x, y, c=colours, s=size)
        
    
        # Lines to show statistical limits
        ax.axhline(-np.log10(1-confidence_interval), color='black', linestyle = '--', zorder=-1)
        if uptake == 'abs':
            ax.axvline(sem, color='black', linestyle = '--', zorder=-1)
            ax.axvline(-sem, color='black', linestyle = '--', zorder=-1)
    
        if annotations:
            for x, y, string in zip(x[results.significant], y[results.significant], np.array(annotations)[results.significant]):
                ax.annotate(string, (x,y), fontsize=fontsize)
    
        
    plt.suptitle('Volcano plot')
    plt.show()
    
    

def _woods_plot(
        all_results: dict,
        sequences,
        sem: float,
        uptake = 'abs',
        colour = 'rel',
        avg_significant: bool = False,
        insignificant_colour: tuple = [0.7, 0.7, 0.7]
        ):
    
    valid = ['rel', 'abs']
    assert (uptake in valid) and (colour in valid),  "must select either 'abs' (actual uptake difference) or 'rel' (relative uptake difference)"
    
    fig, axes = plt.subplots(len(all_results), sharex=True, sharey=True, figsize=(len(all_results)*6, len(all_results)*2))
    seq_bounds = [[s.start, s.end] for s in sequences]
    ymax = np.nanmax([np.nanmax(abs(r.uptake_difference)) for r in all_results.values()]) if uptake=='abs' else np.nanmax([np.nanmax(abs(r.relative_uptake_difference)) for r in all_results.values()])
    
    
    # ylabel
    ylabel = 'Uptake difference (Da)' if uptake == 'abs' else 'Relative uptake difference (%)'
    fig.text(0.1, 0.5, ylabel, va='center', rotation='vertical')
    axes[-1].set(xlabel='Residue')
    xmax = 0
    for ax, (time_label, results) in zip(axes, all_results.items()):
        
        ax.set_title(f'{round(time_label, 2)} min', loc='right')
        ax.set(ylim=(-ymax, ymax))
        
        ys = results.uptake_difference if uptake == 'abs' else results.relative_uptake_difference
        cs = results.raw_colour if colour == 'abs' else results.rel_colour
        cs[~results.significant] = insignificant_colour
        
        for y, c, bounds in zip(ys, cs, seq_bounds):
            ax.plot(bounds, [y]*2, c=c, zorder=-1)
            xmax = np.nanmax(xmax, np.nanmax(bounds))    
        if uptake == 'abs':
            ax.axhline(sem, color='black', linestyle = '--', zorder=-1)
            ax.axhline(-sem, color='black', linestyle = '--', zorder=-1)
    axes[-1].set(xlim=(0, xmax))
        
      
        
        
        
    pass

def do_woods_plot(
        exposure_uptakes: dict,
        sequences,
        exposure_colours: dict,
        uptake_threshold: float,
        average_data: dict,
        ):
    
    labels, uptakes = np.array(list(exposure_uptakes.items())).T
    _colours = np.array(list(exposure_colours.values()))
    colours = copy.deepcopy(_colours)
    avg_data = np.array(list(average_data.values()))
    n = len(labels)
    starts = [s.start for s in sequences]
    ends = [s.end for s in sequences]
    
    mapping = {}
    for i in sequences:
        mapping.update(i.mapping)
    
    all_residues = set(np.arange(0, np.np.nanmax(list(mapping.keys()))))
    mapping_residues = set(list(mapping.keys()))
    missing_residues = sorted(list(all_residues.difference(mapping_residues)))
    
    
    max_uptake = np.np.nanmax([np.np.nanmax(np.abs(arr)) for arr in uptakes])
    padding = 1.05
    
    residues = list(mapping.keys())
    
    fig, axes = plt.subplots(n, 1, sharex=True, sharey=True, figsize=(5*n, 5))
    axes[-1].set_xlabel('Residue')
    
    for ax, label, uptake, col, avg in zip(axes, labels, uptakes, colours, avg_data):
        
        col_sel = np.array([np.prod(c) == 1 for c in col])
        col[col_sel] = np.array([0.7, 0.7, 0.7])
        
        avg *= np.array([s.max_uptake for s in sequences])
        
        ax.plot(list(all_residues), (avg[:-1]), c='purple', alpha=0.5)
        
        ax.set_title(f'{label} min', loc='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set(ylim = (-max_uptake*padding, max_uptake*padding), xlim=(np.min(residues)-1, np.np.nanmax(residues)+1),  ylabel='Uptake difference')
        ax.set_xticks(list(mapping.keys()), minor=True)
        #ax.set_xticklabels(list(mapping.values()), fontdict={'fontsize':4})
       
        ax.axhline(uptake_threshold, color='black', linestyle = '--', zorder=-1, linewidth=1)
        ax.axhline(-uptake_threshold, color='black', linestyle = '--', zorder=-1, linewidth=1)
        ax.set_facecolor('white')
        
        for y, c, x1, x2 in zip(uptake, col, starts, ends):
            ax.plot([x1, x2], [y, y], c=c, zorder=-1)
            
        for missing_res in missing_residues:
            ax.axvspan(missing_res-1, missing_res+1, color='white')
    
    
    plt.tight_layout()
    plt.show()






def do_colourbar(limit):
    
    limit *= 100
    
    reds = cm.Reds(np.linspace(0, 1, 200))
    blues = cm.Blues(np.linspace(1, 0, 200))
    
    colours = np.concatenate([blues, reds], axis=0)
    
    image = [colours for _ in range(20)]
    
    ticks = np.arange(0, int(limit)+1)
    locations = ((ticks / limit) * 200) + 200
    r_locations = 200 - ((ticks / limit) * 200)
    all_loc = np.concatenate([r_locations, locations])
    
    
    
    fig, ax = plt.subplots()
    ax.imshow(image, origin = 'lower')
    ax.set_xticks([0, 200, 400])
    ax.set_xticklabels([round(-limit, 2), 0, round(limit, 2)])
    ax.set_yticks([])
    ax.set_xticks(all_loc, minor= True)
    ax.spines[['top', 'left', 'right']].set_visible(False)
    ax.set(xlabel='Relative uptake difference (%)')
    
    plt.show()
    


class SequenceUptake:
    '''
    Represents HDX results by sequence, containing multiple uptake data 
    for different exposure lengths
    
    Statistical test is constructed depending on user input and given to class
    '''
    def __init__(
            self,
            sequence: str,
            start: int,
            end: int,
            max_uptake: int,
            statistical_test,
            states: dict,
            ):
        
        self.sequence = sequence
        self.start = start
        self.end = end
        self.max_uptake = float(max_uptake)
        self.statistical_test = statistical_test
        self.states = states
        self.results = {}
        self.mapping = {np.arange(self.start, self.end+1)[i]:self.sequence[i] for i in range((self.end-self.start)+1)}
    
    def residue_present(
            self, 
            residue:int
            ) -> bool:
        
        return (residue >= self.start) and (residue <= self.end)
    
    def compare_state_uptake(
            self, 
            exposure_key: str, 
            default_first: bool = True,
        ) -> StatsResult:
        
        # Nested dictionary of states: exposures: data
        states_keys = list(self.states.keys())
        
        # find the index for the given exposure
        exposure_index = np.argwhere(exposure_key == self.states[states_keys[0]].exposure)[0][0] 
        
        if default_first:
            d_index, v_index = 0, 1
        else:
            d_index, v_index = 1, 0
        
        # Retrieve default and variant data for that exposure time
        default = self.states[states_keys[d_index]][exposure_index]
        variant = self.states[states_keys[v_index]][exposure_index]
        
        # Plug into given statistical test
        result, p_value, significant = self.statistical_test(default.uptake, variant.uptake, default.uptake_sd, variant.uptake_sd)
        
            
        return StatsResult(
            uptake_difference=result, 
            relative_uptake_difference=(result / self.max_uptake)*100, 
            p_value=p_value, 
            significant=significant
            )
    
    
    def cumulative(self, stats_test):
        
        # Combine information from all exposures
        keys = list(self.states.keys())
        mean1, mean2 = np.sum(self.states[keys[0]].uptake), np.sum(self.states[keys[1]].uptake)
        sd1, sd2 = pooled_sd(self.states[keys[0]].uptake_sd), pooled_sd(self.states[keys[1]].uptake_sd)
        
        result, p_value, significant = stats_test(mean1, mean2, sd1, sd2)
        
        return StatsResult(
            uptake_difference = result, 
            relative_uptake_difference=result / (self.max_uptake)*100, 
            p_value=p_value, 
            significant=significant
            )
        
def statistical_factory(
        num_repeats: int, 
        confidence: float = 0.95,
        SEM = None
        ):
    '''
    User creation of statistical tests
    '''
    
    # Parameters
    DOF = (num_repeats * 2) - 2
    t_crit = stats.t.ppf(confidence, DOF)
    
    # Significance testing
    p_value = lambda m1, m2, s1, s2: stats.ttest_ind_from_stats(m1, s1, num_repeats, m2, s2, num_repeats).pvalue
    t_significant = lambda m1, m2, s1, s2: p_value(m1, m2, s1, s2) < (1 - confidence)
    global_significant = lambda m1, m2: abs(m1-m2) > SEM * t_crit
    
    # Hybrid add-on
    if SEM is None:
        significant = t_significant
    else:
        significant = lambda m1, m2, s1, s2: t_significant(m1, m2, s1, s2) & global_significant(m1, m2)
    
    # Function factory
    def statistical_test(m1, m2, s1, s2):
        
        uptake_difference = m2 - m1
        p = p_value(m1, m2, s1, s2)
        sig = significant(m1, m2, s1, s2)
        
        return uptake_difference, p, sig
        
    return statistical_test

def make_sequences(
        results: HDXResults,
        statistical_test,
        ) -> List[SequenceUptake]:
    
    ''' 
    Create sequence objects from a results file
    '''
    
    params = results.parameters
    sequences = []
    for index, sequence_data in enumerate(results.parse(params.block_size)):
        
        assert len(set(sequence_data.sequence)) == 1, f'Block contains different sequences, error near row {index*params.block_size}'
        
        state_data = {s.state[0]: s for s in sequence_data.parse(len(params.exposures))}
        sequences.append(
            SequenceUptake(
                sequence = sequence_data.sequence[0],
                start = int(sequence_data.start[0]),
                end = int(sequence_data.end[0]),
                max_uptake = float(sequence_data.maxuptake[0]),
                statistical_test = statistical_test,
                states = state_data
                )
            )
        
    return sequences

class LU:
    def __init__(self):
        self.lu = '/'
    
    
lu = LU()


@cmd.extend
def SAUSC(
        results_file: str = None,
        num_repeats: int = 3,
        confidence_interval: float = 0.95,
        hybrid_test: bool = 'True',
        protection_colour: str = 'Blues',
        deprotection_colour: str = 'Reds',
        insignificant_colour = '[1.0, 1.0, 1.0]',
        no_coverage_colour = '[0.1, 0.1, 0.1]',
        normalise_global: bool = 'True',
        debug_messages: bool = 'False'
        ):
    
    '''
    DESCRIPTION

        Take a HDX results excel comparing a default and variant state, colour significant differences in a gradient manner.
        Colours are normalised either by per exposure setting, or globally.
        You need to enter the correct number of repeats that the data represents as this information isn't saved.
        The specified confidence interval will determine what sequences are considered significant and are coloured in.
        

    USAGE

        SAUSC [ results_file ], [ num_repeats ],

    EXAMPLE
    
        SAUSC your_results_file, normalise_global=True, 
    
    '''
    
    valid_cols = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                  'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                  'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    
    if not protection_colour in valid_cols:
        raise Exception(f'Protection colour: {protection_colour} is not valid!\n\nPlease use one of {valid_cols}')
    
    if not deprotection_colour in valid_cols:
        raise Exception(f'Protection colour: {deprotection_colour} is not valid!\nPlease use one of {valid_cols}')
    
    p_cmap = cm.get_cmap(protection_colour)
    dp_cmap = cm.get_cmap(deprotection_colour)
    
    if results_file is None:
        try:
            # File browser        
            root = tk.Tk()        
            results = filedialog.askopenfilenames(
                parent = root, 
                initialdir = lu.lu, 
                initialfile = 'tmp',
                filetypes = [("CSV", "*.csv"), ("All files", "*")]
                )
            
            results_file = results[0]
            root.destroy()
        
            lu.lu = results_file[:-len(results_file.split('/')[-1])]
        except IndexError:
            print('No file selected')
            root.destroy()
    debug = _check_str_bool(debug_messages)
    
    
    # convert string pymol input to normal
    if debug: 
        print('Reading user input...')
    num_repeats = int(num_repeats)
    confidence_interval = float(confidence_interval)
    insignificant_colour = [float(i) for i in insignificant_colour.strip('[]').split(',')]
    no_coverage_colour = [float(i) for i in no_coverage_colour.strip('[]').split(',')]
    normalise_global = _check_str_bool(normalise_global)
    hybrid_test = _check_str_bool(hybrid_test)
    
    # format confidence interval
    if (confidence_interval <= 100) and (confidence_interval > 1):
        print('Taking confidence interval as 1-100...')
        confidence_interval /= 100
        
    elif (confidence_interval > 0) and (confidence_interval <= 1):
        print('Taking confidence interval as 0-1...')
    else:
        raise TypeError('Invalid confidence interval')
    
    
    # Convert excel to dict
    if debug: 
        print('Converting data to classes...')
    results_dict = HDX_excel_to_dict(results_file)

    # Load into custom recursive parser class
    results = HDXResults(results_dict)
    
    # Pull auto-determined parameters
    parameters = results.parameters
    
    # Calculate global statistical variables
    sem = pooled_standard_error_mean(results.uptake_sd.astype(float), num_repeats)
    t_crit = stats.t.ppf(confidence_interval, (num_repeats * 2) - 2)
    
    # Define statistical test
    args = [num_repeats, confidence_interval]
    if hybrid_test: 
        args.append(sem)
    significance_test = statistical_factory(*args)
    
    # Has to be hybrid
    cumulative_stats_test = statistical_factory(num_repeats, confidence_interval, SEM = sem * len(parameters.exposures - {0.0}))
    
    # Unpack into sequence data
    sequences: list[SequenceUptake] = make_sequences(results, significance_test)
    
    # Calculate results
    exposure_results: dict[float, list[StatsResult]] = {exp: [seq.compare_state_uptake(exp) for seq in sequences] for exp in parameters.exposures - {0.0}}
    cumulative_results: list[StatsResult] = [seq.cumulative(cumulative_stats_test) for seq in sequences]
    
    # Functions to pull data
    get_raw = lambda results: np.array([res.uptake_difference for res in results])
    get_rel = lambda results: np.array([res.relative_uptake_difference for res in results])
    get_significant = lambda results: np.array([res.significant for res in results])
   
    
    # Quick tool for converting to colours based on colormaps
    def make_colours(
            results: np.ndarray, 
            norm_value: float,
            significant: np.ndarray = None,
            )-> np.ndarray:
        
        normalised_values = results / norm_value
        
        positive = results > 0 
        negative = results < 0
        no_data = np.isnan(results)
        
        colours = np.full((len(results), 3), insignificant_colour, float)
        if significant is None:
            significant = np.full(len(results), fill_value=True, dtype=bool)
        
        
        if no_data.any():
            colours[no_data] = np.full((no_data.sum(), 3), no_coverage_colour, float)
        if positive.any():
            colours[positive & significant] = dp_cmap(abs(normalised_values[positive & significant]))[:, :-1]
        if negative.any():
            colours[negative & significant] = p_cmap(abs(normalised_values[negative & significant]))[:, :-1]
        print([(r,norm, c, s, p, n) for r, norm, c, s, p, n in zip(results, normalised_values, colours, significant, positive, negative)])
        return colours
    
    
    
    
    # Colouring
    
    # max value always taken from exposure results
    # Cumulative values will always be less
    
    if normalise_global:

        max_exposure = lambda f: np.nanmax([np.nanmax(abs(f(array))) for array in exposure_results.values()])
        
        for results in exposure_results.values():
            raw_colours = make_colours(get_raw(results), significant = get_significant(results), norm_value = max_exposure(get_raw))
            rel_colours = make_colours(get_rel(results), significant = get_significant(results), norm_value = max_exposure(get_rel))
            
            for res, raw, rel in zip(results, raw_colours, rel_colours):
                res.raw_colour = raw
                res.rel_colour = rel
            
    else:
        max_exposure = lambda f: {exp: np.nanmax(abs(f(exposure_results[exp]))) for exp in exposure_results.keys()}
        raw_norms = max_exposure(get_raw)
        rel_norms = max_exposure(get_rel)
        for results, raw_norm, rel_norm in zip(exposure_results.values(), raw_norms.values(), rel_norms.values()):
            raw_colours = make_colours(get_raw(results), significant=get_significant(results), norm_value = raw_norm)
            rel_colours = make_colours(get_rel(results), significant=get_significant(results), norm_value = rel_norm)
            for res, raw, rel in zip(results, raw_colours, rel_colours):
                res.raw_colour = raw
                res.rel_colour = rel
    
    # Cumulative must be done on global norm...
    cumulative_max_exposure = lambda f: np.nanmax(abs(f(cumulative_results)))
    cumulative_raw_colours = make_colours(get_raw(cumulative_results), significant = get_significant(cumulative_results), norm_value = cumulative_max_exposure(get_raw))
    cumulative_rel_colours = make_colours(get_rel(cumulative_results), significant = get_significant(cumulative_results), norm_value = cumulative_max_exposure(get_rel))
    
    for res, raw, rel in zip(cumulative_results, cumulative_raw_colours, cumulative_rel_colours):
        res.raw_colour = raw
        res.rel_colour = rel
    
    
    
    # SOmewhere get the averaged data
    # Only done on raw uptake difference, can't be done on relative difference as different sizes!
    
    residues: dict[float, list[Residue]] = {exp: residue_averaged_exposure(sequences, StatsResultsWrapper(v), significant_only=True) for exp, v in exposure_results.items()}
    cumulative_residues: list[Residue] = residue_averaged_exposure(sequences, StatsResultsWrapper(cumulative_results), significant_only=True)
    
    glob_norm = np.nanmax([np.nanmax(abs(get_raw(array))) for array in exposure_results.values()])
    
    if normalise_global:
        residue_colours = {
            exp: make_colours(
                results = np.array([i.mean for i in residues[exp]]), 
                norm_value = glob_norm,
                significant=np.array([i.mean != 0 for i in residues[exp]])) 
            for exp, r in exposure_results.items()
            }
    else:
        residue_colours = {
            exp: make_colours(
                results = np.array([i.mean for i in residues[exp]]), 
                norm_value = raw_norms[exp],
                significant=np.array([i.mean != 0 for i in residues[exp]])) 
            for (exp, r) in exposure_results.items()
            }
        
    
    cumulative_colours = make_colours(np.array([i.mean for i in cumulative_residues]), norm_value = np.nanmax(np.abs([i.mean for i in cumulative_residues])), significant=np.array([i.mean!=0 for i in cumulative_residues]))
    
    # Put colour into object
    for exposure in residues.keys():
        for r, c in zip(residues[exposure], residue_colours[exposure]):
            r.colour = c

    for r,c in zip(cumulative_residues, cumulative_colours):
        r.colour = c
    
    @cmd.extend
    def colourbar():
        pass
        
    
    @cmd.extend
    def volcano_plot(
        cumulative = 'False', 
        annotate = 'False'    ,
        uptake: str = 'abs',
        colour: str = 'rel',
        xpadding = '1.05',
        insignificant_colour: tuple = '[0.7, 0.7, 0.7]',
        size = '20',
        fontsize: int = '5', 
    ):
        
        # Check variables
        cumulative = _check_str_bool(cumulative)
        annotate = _check_str_bool(annotate)
        xpadding = float(xpadding)
        insignificant_colour = [float(i) for i in insignificant_colour.strip('[]').split(',')]
        size = float(size)
        fontsize = float(fontsize)
        
        # Pull data needed
        anno = [f'{s.start} - {s.end}' for s in sequences] if annotate else None
        data = StatsResultsWrapper(cumulative_results) if cumulative else {k:StatsResultsWrapper(v) for k,v in sorted(exposure_results.items())}
        plot_sem =  sem * t_crit * len(exposure_results) if cumulative else sem * t_crit
        
        # Define function
        func = _cumulative_volcano_plot if cumulative else _volcano_plot
        
        # Run function
        try:
            func(
                data,
                sem = plot_sem,
                confidence_interval = confidence_interval,
                annotations = anno,
                uptake = uptake,
                colour = colour,
                xpadding = xpadding,
                insignificant_colour = insignificant_colour,
                size = size,
                fontsize = fontsize,
                )
        except BaseException as e:
            print(f"Failed to do volcano plot: {e}")

        plt.tight_layout()
        plt.show()
    
        
        
    # TODO all the other plots
    '''
    @cmd.extend
    def woods_plot():
        
        do_woods_plot(
            exposure_uptakes = exposure_data_all, 
            sequences = sequences, 
            exposure_colours = sequence_colours, 
            uptake_threshold = sem * t_crit,
            average_data = woods_plot_avgs
            )
        
    @cmd.extend
    def all_plots(annotate= 'False'):
        anno = _check_str_bool(annotate)
        
        if anno:
            anno = [f'{s.start} - {s.end}' for s in sequences]
        
        do_volcano_plot(
            exposure_uptakes = exposure_data_all, 
            exposure_colours = sequence_colours, 
            p_values = p_values, 
            uptake_threshold = sem * t_crit, 
            confidence = confidence_interval,
            annotations = anno
            )
        
        
        do_cumulative_volcano_plot(cum_dif, cum_p, cum_colours, sem * len(parameters.exposures - {0.0}) * t_crit, confidence_interval, anno)   
    
    
            
        do_woods_plot(
            exposure_uptakes = exposure_data_all, 
            sequences = sequences, 
            exposure_colours = sequence_colours, 
            uptake_threshold = sem * t_crit,
            average_data = exposure_data_residue
            )
        
        do_colourbar(max_max)
    
    
    '''
    
    
    ERROR_COLOUR = (204, 0, 153)
    
    all_colours = [tuple(c) for i in residue_colours.values() for c in i] + [tuple(c) for c in cumulative_colours] + [ERROR_COLOUR]
    
    
    unique_cols = set(all_colours)
    
    pymol_cols = {f'custom{np.random.random()}': list(col) for col in unique_cols}
    cols_to_name = {tuple(c):name for name, c in pymol_cols.items()}
    

    # Set colours by string in pymol, I think that's needed...
    print('Setting colours...')
    for name, colour in pymol_cols.items():
        cmd.set_color(name=name, rgb=colour)
    
    for exposure in sorted(parameters.exposures - {0.0}):
          
        print('Reverting colours in case we have missed anything...')
        for r in cumulative_residues:
            cmd.color(cols_to_name[ERROR_COLOUR], selection=f'res {r.n}')

        print('Colouring by residue...')
        for r in residues[exposure]:
            print(f'res{r.n} = {r.colour}')
            cmd.color(cols_to_name[tuple(r.colour)], selection=f'res {r.n}')
        

        # Set the scene
        normalise = 'across exposures' if normalise_global == True else 'per exposure'
        message = f'Exposure {exposure}, normalised {normalise}\nConfidence {confidence_interval*100}%'
        
        cmd.scene(key="new", action='store', message=message, color=1)
      
        
        print('Saving scene...')
        
        


    # Colour the cumulative plot as a new scene
    print('Reverting colours in case we have missed anything...')
    for r in cumulative_residues:
        cmd.color(cols_to_name[ERROR_COLOUR], selection=f'res {r.n}')

    
    print('Colouring by residue...')
    for r in cumulative_residues:
        print(f'res {r.n} = {r.colour}')
        cmd.color(cols_to_name[tuple(colour)], selection=f'res {r.n}')

    

    message = f'Cumulative uptake across exposures, normalised to global maximum\nConfidence {confidence_interval*100}%'
    cmd.scene(key='new', action='store', message=message, color=1)




print('\n\nSuccesfully loaded SAUSC!\n\nParameters and defaults:\nnum_repeats  -  3\nconfidence_interval  -  0.95\nhybrid_test  -  True\nprotection_colour  -  Blues\ndeprotection_colour  -  Reds\ninsignificant_colour  -  [1.0, 1.0, 1.0]\nno_coverage_colour  -  [0.1, 0.1, 0.1]\nnormalise_global  -  True\ndebug_messages  -  False\n\nUsage: Just type "SAUSC" into the command prompt to navigate to a results file and use the default settings shown above!')





