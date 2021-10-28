#!/usr/bin/env python
"""
Nice publication quality figures. See examples for usage.
"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

cm_in_inches = 0.393701
golden_ratio = 1.61803398875

plt.rc("pgf", texsystem="lualatex")
plt.rc("pgf", preamble=
       '\usepackage{amsmath},'
       '\usepackage{yfonts},'
       '\usepackage[T1]{fontenc},'
       '\usepackage{txfonts},'
       '\usepackage{fontspec},'
       '\usepackage[Symbolsmallscale]{upgreek},'
       '\usepackage{times},'
       '\usepackage{blindtext}')
plt.rc('font', **{'family':'serif', 'serif':['Times'], 'size': 10.0})
plt.rc('lines', linewidth=1.0)
plt.rc('axes', linewidth=0.5)
plt.rc('xtick', labelsize='medium', direction='in')
plt.rc('ytick', labelsize='medium', direction='in')
plt.rc('xtick.major', size=4.0, width=0.5)
plt.rc('xtick.minor', size=2.0, width=0.5)
plt.rc('ytick.major', size=4.0, width=0.5)
plt.rc('ytick.minor', size=2.0, width=0.5)
plt.rc('legend', fontsize='small', loc='best')
plt.rc('text', usetex=True)

# Custom diverging colormap for white background:
cdict_blrddark = {
			'red':   ((0.0, 0.0, 0.0),
                     (0.5, 0.0, 0.0),
                     (1.0, 1.0, 1.0)),

			'green': ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0)),

			'blue':  ((0.0, 0.0, 1.0),
                     (0.5, 0.0, 0.0),
                     (1.0, 0.0, 0.0))
			}
cdict_rdbldark = {
			'red':   ((0.0, 1.0, 1.0),
                     (0.5, 0.0, 0.0),
                     (1.0, 0.0, 0.0)),

			'green': ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0)),

			'blue':  ((0.0, 0.0, 0.0),
                     (0.5, 0.0, 0.0),
                     (1.0, 1.0, 1.0))
			}
matplotlib.cm.register_cmap(name="BlRdDark", data=cdict_blrddark)
matplotlib.cm.register_cmap(name="RdBlDark", data=cdict_rdbldark)

#plotting:
def filled_plot(ax, x, y1, y2, linestyle ="solid", alpha=0.5, linealpha=1.0, linewidth=1.0, label=None):
    base_line, = ax.plot(x, y1, linestyle=linestyle, linewidth=linewidth, label=label, alpha=linealpha)
    color = base_line.get_color()
    ax.plot(x, y2, color=color, linestyle=linestyle, linewidth=linewidth, alpha=linealpha)
    ax.fill_between(x, y1, y2, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)

def filled_error_plot(ax, x, y, err, linestyle ="solid", alpha=0.5, linealpha=1.0, linewidth=1.0, label=None):
    filled_plot(ax, x, y-err, y+err, linestyle, alpha, linealpha, linewidth, label)

def create_figure(width_inches=None, height_inches=None, height_multiplier=None):
    fig = plt.figure()
    fig.horizontal_merged = False
    fig.vertical_merged = False
    if width_inches is None:
        width_inches = 8.6 * cm_in_inches
    fig.default_height = width_inches / golden_ratio
    if height_inches is None:
        height_inches = fig.default_height
    if height_multiplier is not None:
        height_inches *= height_multiplier
    fig.set_size_inches(width_inches, height_inches)
    return fig

def set_default_spacing(fig):
    # These are a good start in most cases, but may require some manual adjustment.
    # fig.subplots_adjust(bottom=0.2 * fig.default_height / fig.get_size_inches()[1]) # default
    fig.subplots_adjust(bottom=0.18 * fig.default_height / fig.get_size_inches()[1])
    fig.subplots_adjust(left=0.18) # default
    # fig.subplots_adjust(left=0.165)
    fig.subplots_adjust(top=1.0 - 0.05 * fig.default_height / fig.get_size_inches()[1]) # default
    # fig.subplots_adjust(top=1.0 - 0.01 * fig.default_height / fig.get_size_inches()[1])
    fig.subplots_adjust(right=0.95)

def create_single_panel(fig, xlabel=None, ylabel=None, palette='Set1', numcolors=9):
    ax = fig.add_subplot(111)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        fig.subplots_adjust(bottom=0.2)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_prop_cycle('color', plt.get_cmap(palette)(np.linspace(0,1,numcolors)))
    set_default_spacing(fig)

    return ax

# This is a 2x1 plot, with the axes merged by default.
# TODO: Can be generalized to Nx1.
def create_vertical_split(fig, merged=True, xlabel=None, ylabel=None, palette=('Set1', 'Set1'), numcolors=(9, 9)):
    ax1 = fig.add_subplot(121)

    if merged:
        fig.vertical_merged = True
        ax2 = fig.add_subplot(122, sharey=ax1) # Share axes.
        fig.subplots_adjust(wspace=0) # Merge axes.
        plt.setp(\
            [a.get_yticklabels() \
             for a in fig.axes[1:]], visible=False) # Remove ticks from right axes.
        if ylabel is not None:
            ax1.set_ylabel(ylabel[0]) # No ylabel for right axes.
    else:
        fig.vertical_merged = False
        ax2 = fig.add_subplot(122)
        if ylabel is not None:
            ax1.set_ylabel(ylabel[0])
            ax2.set_ylabel(ylabel[1])
        fig.subplots_adjust(wspace=0.5) # default
    if xlabel is not None:
        ax1.set_xlabel(xlabel[0])
        ax2.set_xlabel(xlabel[1])

    ax1.set_prop_cycle('color', plt.get_cmap(palette[0])(np.linspace(0,1,numcolors[0])))
    ax2.set_prop_cycle('color', plt.get_cmap(palette[1])(np.linspace(0,1,numcolors[1])))
    set_default_spacing(fig)
    return ax1, ax2

# This is a 1x2 plot, with the axes merged by default.
# TODO: Can be generalized to 1xN.
def create_horizontal_split(fig, merged=True, xlabel=None, ylabel=None, palette=('Set1', 'Set1'), numcolors=(9, 9)):
    ax1 = fig.add_subplot(211)

    if merged:
        fig.horizontal_merged = True
        ax2 = fig.add_subplot(212, sharex=ax1) # Share axes.
        # ax2 = fig.add_subplot(212) # Share axes.
        fig.subplots_adjust(hspace=0) # Merge axes.
        plt.setp(\
            [a.get_xticklabels() \
             for a in fig.axes[:-1]], visible=False) # Remove ticks from top axes.
        if xlabel is not None:
            ax2.set_xlabel(xlabel[1]) # No ylabel for top axes.
    else:
        fig.horizontal_merged = False
        ax2 = fig.add_subplot(212)
        if ylabel is not None:
            ax1.set_xlabel(xlabel[0])
            ax2.set_xlabel(xlabel[1])
        # fig.subplots_adjust(hspace=0.5) # default
        fig.subplots_adjust(hspace=0.35)
    if ylabel is not None:
        ax1.set_ylabel(ylabel[0])
        ax2.set_ylabel(ylabel[1])

    ax1.set_prop_cycle('color', plt.get_cmap(palette[0])(np.linspace(0.2,1,numcolors[0])))
    ax2.set_prop_cycle('color', plt.get_cmap(palette[1])(np.linspace(0.2,1,numcolors[1])))
    set_default_spacing(fig)
    return ax1, ax2

# This is a 2x2 plot, with the axes always merged.
# TODO: Can be generalized to NxM, partially merged.
def create_quad_split(fig, xlabel=None, ylabel=None, palette=('Set1', 'Set1', 'Set1', 'Set1'), numcolors=(9, 9, 9, 9)):
    fig.horizontal_merged = True
    fig.vertical_merged = True
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, sharey=ax1)
    ax3 = fig.add_subplot(223, sharex=ax1)
    ax4 = fig.add_subplot(224, sharey=ax3)

    # Merge axes.
    fig.subplots_adjust(wspace=0)
    fig.subplots_adjust(hspace=0)

    # Remove ticks from top axes.
    for p in [0, 1]:
        plt.setp(fig.axes[p].get_xticklabels(), visible=False)
    # Remove ticks from right axes.
    for p in [1, 3]:
        plt.setp(fig.axes[p].get_yticklabels(), visible=False)

    # Set axis labels.
    if xlabel is not None:
        ax3.set_xlabel(xlabel[0])
        ax4.set_xlabel(xlabel[1])
    if ylabel is not None:
        ax1.set_ylabel(ylabel[0])
        ax3.set_ylabel(ylabel[1])

    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.set_prop_cycle('color', plt.get_cmap(palette[i])(np.linspace(0,1,numcolors[i])))
    set_default_spacing(fig)
    return ax1, ax2, ax3, ax4


def finalize_and_save(fig, filename='plot.pdf'):
    axes = fig.get_axes()
    for ax in axes:
        # legend = ax.legend(loc='best', handlelength=1.25, ncol=2, fancybox=True, framealpha=0.8)
        # legend = ax.legend(loc='upper right', handlelength=1.0, ncol=2, fancybox=True, framealpha=0.8)
        legend = ax.legend(loc='upper right', handlelength=1.0, ncol=3, fancybox=True, framealpha=0.8)
        if legend is not None:
            legend.get_frame().set_linewidth(0.5)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.margins(0)
    if fig.vertical_merged and not fig.horizontal_merged:
        plt.setp(\
            [a.get_xticklabels()[-1] \
             for a in fig.axes[:-1]], visible=False) # Remove last label from left axes.
    if fig.horizontal_merged and not fig.vertical_merged:
        plt.setp(\
            [a.get_yticklabels()[-1] \
             for a in fig.axes[1:]], visible=False) # Remove last label from bottom axes.
    if fig.horizontal_merged and fig.vertical_merged:
        plt.setp(fig.axes[2].get_yticklabels()[-1], visible=False)
        plt.setp(fig.axes[2].get_xticklabels()[-1], visible=False)
    fig.savefig(filename, dpi=400)
