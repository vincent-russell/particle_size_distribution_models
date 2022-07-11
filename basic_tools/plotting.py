"""
Plotting functions
"""


#######################################################
# Modules:
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

# Local modules:
from basic_tools.miscellaneous import get_kwarg_value


#######################################################
# Plot 1D functions f(x) over domain [a, b] with number of points Nplot (default is 200):
def plot_1D(f, a, b, **kwargs):

    #######################
    # Personal parameters:
    location = get_kwarg_value(kwargs, 'location', 'Middle')  # Plotting for uni office or elsewhere

    #######################
    # Universal parameters:
    Nplot = get_kwarg_value(kwargs, 'Nplot', 200)  # Number of points for plot
    title = get_kwarg_value(kwargs, 'title', '')  # Title for plot
    xdiscrete = get_kwarg_value(kwargs, 'xdiscrete', None)  # Discretisation for plotting on x-axis
    xticks = get_kwarg_value(kwargs, 'xticks', None)  # Plot x-tick labels
    xscale = get_kwarg_value(kwargs, 'xscale', 'linear')  # x-axis scaling ('linear' or 'log')
    yscale = get_kwarg_value(kwargs, 'yscale', 'linear')  # y-axis scaling ('linear' or 'log')
    xlabel = get_kwarg_value(kwargs, 'xlabel', 'x')  # x-axis label
    ylabel = get_kwarg_value(kwargs, 'ylabel', 'y')  # y-axis label
    ylabelrotation = get_kwarg_value(kwargs, 'ylabelrotation', 0)  # y-axis label rotation from axis
    ylabelpad = get_kwarg_value(kwargs, 'ylabelpad', 15)  # y-axis label spacing from axis
    xfontsize = get_kwarg_value(kwargs, 'xfontsize', 16)  # x-axis label font size
    yfontsize = get_kwarg_value(kwargs, 'yfontsize', 16)  # y-axis label font size
    titlefontsize = get_kwarg_value(kwargs, 'titlefontsize', 16)  # Title font size
    style = get_kwarg_value(kwargs, 'style', 'default')  # Style of figure
    doing_mainloop = get_kwarg_value(kwargs, 'doing_mainloop', 'True')  # Automatically does tk.mainloop(); (set 'False' if plotting multiple GUI / figures)

    #######################
    # Changing Nplot if directly adding discretisation for x-axis
    if xdiscrete is not None:
        Nplot = len(xdiscrete)

    #######################
    # Computing discretisation and corresponding function values:
    x = np.linspace(a, b, Nplot)  # Computing x-values
    y = np.zeros(Nplot)  # Initialising function values
    for i in range(Nplot):
        y[i] = f(x[i])  # Computing function values

    #######################
    # Limit parameters:
    xlimits = get_kwarg_value(kwargs, 'xlimits', [a, b])  # Plot boundary limits for x-axis (invalid if using specified x discretisation)
    ylimits = get_kwarg_value(kwargs, 'ylimits', [np.min(y), np.max(y)])  # Plot boundary limits for y-axis

    #######################
    # Creating new figure on tkinter GUI:
    root = tk.Tk()  # Creating root (backend) of GUI
    plt.style.use(style)  # Setting style of figure

    # Creating figure with specified window size and position:
    if location == 'Uni' or location == 'Uni1':
        fig = plt.Figure(figsize=(9.05, 7.65), dpi=100)  # Creating figure
        root.geometry('905x795-2670+290')  # At university office
    elif location == 'Uni2':
        fig = plt.Figure(figsize=(8.70, 6.90), dpi=100)  # Creating figure
        root.geometry('870x720+2040-582')  # At university office
    elif location == 'Uni3':
        fig = plt.Figure(figsize=(8.70, 6.90), dpi=100)  # Creating figure
        root.geometry('870x720+2040+562')  # At university office
    elif location == 'Home' or location == 'Home1':
        fig = plt.Figure(figsize=(6.6, 5.6), dpi=100)  # Creating figure
        root.geometry('660x560-3170+605')  # At home
    elif location == 'Home2':
        fig = plt.Figure(figsize=(6.6, 5.7), dpi=100)  # Creating figure
        root.geometry('660x570+3170-590')  # At home
    elif location == 'Home3':
        fig = plt.Figure(figsize=(6.6, 5.6), dpi=100)  # Creating figure
        root.geometry('660x560+3170+605')  # At home
    elif location == 'Presentation' or location == 'Presentation1' or location == 'Presentation2' or location == 'Presentation3':
        fig = plt.Figure(figsize=(6.50, 5.00), dpi=100)  # Creating figure
        root.geometry('650x530-100+100')  # Default is middle of primary screen
    else:
        fig = plt.Figure(figsize=(10.00, 9.00), dpi=100)  # Creating figure
        root.geometry('1000x930-100+100')  # Default is middle of primary screen

    # Creating new figure and axes:
    ax = fig.add_subplot(111)  # Creating subplot in figure
    if xdiscrete is not None:  # If plotting with specified x discretisation
        ax.plot(xdiscrete, y)  # Creating plot
        ax.set_xlim([xdiscrete[0], xdiscrete[-1]])  # Sets limits in x-axis
    else:  # If plotting with [a, b] discretisation
        ax.plot(x, y)  # Creating plot
        ax.set_xlim(xlimits)  # Sets limits in x-axis
    ax.set_ylim(ylimits)  # Sets limits in y-axis
    ax.set_xscale(xscale)  # Sets x-axis to log or linear
    ax.set_yscale(yscale)  # Sets y-axis to log or linear
    ax.set_xlabel(xlabel, fontsize=xfontsize)  # Adds xlabel
    ax.set_ylabel(ylabel, fontsize=yfontsize, rotation=ylabelrotation, labelpad=ylabelpad)  # Adds ylabel
    ax.set_title(title, fontsize=titlefontsize)  # Adds title
    if xticks is not None:
        plt.setp(ax, xticks=xticks, xticklabels=xticks)  # Modifies x-tick labels
    ax.grid()  # Adds a grid to the figure

    # Creating canvas and adding figure to canvas and GUI, with NavigationToolbar:
    canvas = FigureCanvasTkAgg(fig, master=root)  # Adding canvas
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)  # Creating space for figure and toolbar on canvas
    NavigationToolbar2Tk(canvas, root)  # Adding navigation toolbar to figure

    if doing_mainloop:  # Automatically does mainloop() to plot GUI (turn this off if plotting multiple GUI / figures)
        tk.mainloop()


#######################################################
# Surface plot of a 2D array (i.e. matrix); works only for square matrices:
def plot_matrix(matrix, **kwargs):

    #######################
    # Personal parameters:
    location = get_kwarg_value(kwargs, 'location', 'Middle')  # Plotting for uni office or elsewhere

    #######################
    # Universal parameters:
    title = get_kwarg_value(kwargs, 'title', '')  # Title for plot
    xlabel = get_kwarg_value(kwargs, 'xlabel', 'x')  # x-axis label
    ylabel = get_kwarg_value(kwargs, 'ylabel', 'y')  # y-axis label
    xfontsize = get_kwarg_value(kwargs, 'xfontsize', 16)  # x-axis label font size
    yfontsize = get_kwarg_value(kwargs, 'yfontsize', 16)  # y-axis label font size
    titlefontsize = get_kwarg_value(kwargs, 'titlefontsize', 16)  # Title font size
    style = get_kwarg_value(kwargs, 'style', 'default')  # Style of figure
    doing_mainloop = get_kwarg_value(kwargs, 'doing_mainloop', 'True')  # Automatically does tk.mainloop(); (set 'False' if plotting multiple GUI / figures)

    #######################
    # Computing meshgrid for axes:
    N = len(matrix)  # Dimension of matrix
    x = y = np.arange(0, N, 1)  # Creates arrays for x and y axes
    X, Y = np.meshgrid(x, y)  # Creates coordinates (x, y) axes for plotting subplot

    #######################
    # Creating new figure on tkinter GUI:
    root = tk.Tk()  # Creating root (backend) of GUI
    plt.style.use(style)  # Setting style of figure

    # Creating figure with specified window size and position:
    if location == 'Uni' or location == 'Uni1':
        fig = plt.Figure(figsize=(9.05, 7.65), dpi=100)  # Creating figure
        root.geometry('905x795-2670+290')  # At university office
    elif location == 'Uni2':
        fig = plt.Figure(figsize=(8.70, 6.90), dpi=100)  # Creating figure
        root.geometry('870x720+2040-582')  # At university office
    elif location == 'Uni3':
        fig = plt.Figure(figsize=(8.70, 6.90), dpi=100)  # Creating figure
        root.geometry('870x720+2040+562')  # At university office
    elif location == 'Home' or location == 'Home1':
        fig = plt.Figure(figsize=(7.2, 5.07), dpi=100)  # Creating figure
        root.geometry('725x537-2187-124')  # At home
    elif location == 'Home2':
        fig = plt.Figure(figsize=(8.70, 6.90), dpi=100)  # Creating figure
        root.geometry('725x537-2187-124')  # At home
    elif location == 'Home3':
        fig = plt.Figure(figsize=(8.70, 6.90), dpi=100)  # Creating figure
        root.geometry('725x537-2187-124')  # At home
    else:
        fig = plt.Figure(figsize=(10.00, 9.00), dpi=100)  # Creating figure
        root.geometry('1000x930-100+100')  # Default is middle of primary screen

    # Creating new figure and axes:
    ax = fig.add_subplot(111, projection='3d')  # Creating subplot in figure with 3-D axes
    ax.plot_surface(X, Y, matrix)  # Plotting 3-D surface
    ax.set_xlabel(xlabel, fontsize=xfontsize)  # Adds xlabel
    ax.set_ylabel(ylabel, fontsize=yfontsize)  # Adds ylabel
    ax.set_title(title, fontsize=titlefontsize)  # Adds title

    # Creating canvas and adding figure to canvas and GUI, with NavigationToolbar:
    canvas = FigureCanvasTkAgg(fig, master=root)  # Adding canvas
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)  # Creating space for figure and toolbar on canvas
    NavigationToolbar2Tk(canvas, root)  # Adding navigation toolbar to figure

    if doing_mainloop:  # Automatically does mainloop() to plot GUI (turn this off if plotting multiple GUI / figures)
        tk.mainloop()


#######################################################
# 1D animation plot through time:
def plot_1D_animation(x, *y, **kwargs):

    #######################
    # Personal parameters:
    location = get_kwarg_value(kwargs, 'location', 'Middle')  # Plotting for uni office or elsewhere

    #######################
    # Universal parameters:
    x_add, Y_add = get_kwarg_value(kwargs, 'plot_add', (None, None))  # Tuple plot_add = (x, Y) where animation plots (x, Y[:, i]) for i = 0, 1, ...
    delay = get_kwarg_value(kwargs, 'delay', 15)  # Delay between frames in milliseconds
    xlimits = get_kwarg_value(kwargs, 'xlimits', [min(x), max(x)])  # Plot boundary limits for x-axis
    ylimits = get_kwarg_value(kwargs, 'ylimits', [np.min(y[0]), np.max(y[0])])  # Plot boundary limits for y-axis
    xticks = get_kwarg_value(kwargs, 'xticks', None)  # Plot x-tick labels
    xticklabels = get_kwarg_value(kwargs, 'xticklabels', xticks)  # Plot x-tick labels
    yticks = get_kwarg_value(kwargs, 'yticks', None)  # Plot y-tick labels
    xscale = get_kwarg_value(kwargs, 'xscale', 'linear')  # x-axis scaling ('linear' or 'log')
    yscale = get_kwarg_value(kwargs, 'yscale', 'linear')  # y-axis scaling ('linear' or 'log')
    xlabel = get_kwarg_value(kwargs, 'xlabel', '')  # x-axis label
    ylabel = get_kwarg_value(kwargs, 'ylabel', '')  # y-axis label
    xfontsize = get_kwarg_value(kwargs, 'xfontsize', 15)  # x-axis label font size
    yfontsize = get_kwarg_value(kwargs, 'yfontsize', 13)  # y-axis label font size
    ylabelrotation = get_kwarg_value(kwargs, 'ylabelrotation', 0)  # y-axis label rotation from axis
    ylabelpad = get_kwarg_value(kwargs, 'ylabelpad', 15)  # y-axis label spacing from axis
    ylabeltop = get_kwarg_value(kwargs, 'ylabeltop', True)  # Set to True for y-axis label to be at top, otherwise at center
    anim_title = get_kwarg_value(kwargs, 'title', '')  # Title for plot
    titlefontsize = get_kwarg_value(kwargs, 'titlefontsize', 15)  # Title font size
    style = get_kwarg_value(kwargs, 'style', 'default')  # Style of figure
    time = get_kwarg_value(kwargs, 'time', None)  # Array where time[i] is plotted (and animated)
    timetext = get_kwarg_value(kwargs, 'timetext', None)  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    timetextfontsize = get_kwarg_value(kwargs, 'timetextfontsize', 13)  # Animated text font size
    timepos = get_kwarg_value(kwargs, 'timepos', (0.6, 0.75))  # Tuple where text position is (x, y) where (0, 0) is bottom left and (1, 1) is top right
    doing_mainloop = get_kwarg_value(kwargs, 'doing_mainloop', True)  # Automatically does tk.mainloop(); (set 'False' if plotting multiple GUI / figures)
    save_ani = get_kwarg_value(kwargs, 'save_ani', False)  # Save animation as .mp4 file
    save_name = get_kwarg_value(kwargs, 'save_name', 'plot_1D_animation')  # Saved animation filename
    save_fps = get_kwarg_value(kwargs, 'save_fps', 60)  # Saved animation fps
    use_widget = get_kwarg_value(kwargs, 'use_widget', False)  # Set to True to add widget to figures

    # Legend labels for lines in plot:
    use_legend = False  # Default
    legend = []  # Initialising
    num_legend = 0  # Initialising
    legend_position = get_kwarg_value(kwargs, 'legend_position', 'upper right')
    if 'legend' in kwargs:
        legend = kwargs['legend']
        # Checking if adding additional plot with legend:
        if x_add is None:
            num_legend = len(legend)
        else:
            num_legend = len(legend) - 1
        use_legend = True

    # Line colors in plot:
    use_color = False  # Default
    line_color = None  # Default
    if 'line_color' in kwargs:
        line_color = get_kwarg_value(kwargs, 'line_color', None)  # List for colors of lines to plot
        use_color = True

    # Line colors in plot:
    use_style = False  # Default
    line_style = None  # Default
    if 'line_style' in kwargs:
        line_style = get_kwarg_value(kwargs, 'line_style', None)  # List for styles of lines to plot
        use_style = True

    #######################
    # Creating new figure on tkinter GUI:
    root = tk.Tk()  # Creating root (backend) of GUI
    plt.style.use(style)  # Setting style of figure

    # Creating figure with specified window size and position:
    if location == 'Uni' or location == 'Uni1':
        fig = plt.Figure(figsize=(9.05, 7.65), dpi=100)  # Creating figure
        root.geometry('905x795-2670+290')  # At university office
    elif location == 'Uni2':
        fig = plt.Figure(figsize=(8.70, 6.90), dpi=100)  # Creating figure
        root.geometry('870x720+2040-582')  # At university office
    elif location == 'Uni3':
        fig = plt.Figure(figsize=(8.70, 6.90), dpi=100)  # Creating figure
        root.geometry('870x720+2040+562')  # At university office
    elif location == 'Home' or location == 'Home1':
        fig = plt.Figure(figsize=(5.18, 4.5), dpi=80)  # Creating figure
        root.geometry('518x450-2547+480')  # At home
    elif location == 'Home2':
        fig = plt.Figure(figsize=(5.18, 4.5), dpi=80)  # Creating figure
        root.geometry('518x450+2483-470')  # At home
    elif location == 'Home3':
        fig = plt.Figure(figsize=(5.18, 4.5), dpi=80)  # Creating figure
        root.geometry('518x450+2483+480')  # At home
    elif location == 'Presentation' or location == 'Presentation1' or location == 'Presentation2' or location == 'Presentation3':
        fig = plt.Figure(figsize=(6.50, 5.00), dpi=100)  # Creating figure
        root.geometry('650x530-100+100')  # Default is middle of primary screen
    else:
        fig = plt.Figure(figsize=(10.00, 9.00), dpi=100)  # Creating figure
        root.geometry('1000x930-100+100')  # Default is middle of primary screen

    # Creating canvas and adding figure to canvas and GUI, with NavigationToolbar:
    canvas = FigureCanvasTkAgg(fig, master=root)  # Adding canvas
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)  # Creating space for figure and toolbar on canvas
    if use_widget:
        NavigationToolbar2Tk(canvas, root)  # Adding navigation toolbar to figure

    # Setting axis variables:
    ax = fig.add_subplot(111)  # Creating subplot in figure
    ax.set_xlim(xlimits)  # Sets limits in x-axis
    ax.set_ylim(ylimits)  # Sets limits in y-axis
    ax.set_xscale(xscale)  # Sets x-axis to log or linear
    ax.set_yscale(yscale)  # Sets y-axis to log or linear
    ax.set_xlabel(xlabel, fontsize=xfontsize)  # Adds xlabel
    ax.set_ylabel(ylabel, fontsize=yfontsize, rotation=ylabelrotation, labelpad=ylabelpad)  # Adds ylabel
    if ylabeltop:
        ax.yaxis.set_label_coords(0.03, 1.02)
    ax.set_title(anim_title, fontsize=titlefontsize)  # Adds title
    if xticks is not None:
        plt.setp(ax, xticks=xticks, xticklabels=xticklabels)  # Modifies x-tick labels
    if yticks is not None:
        plt.setp(ax, yticks=yticks, yticklabels=yticks)  # Modifies y-tick labels
    ax.grid()  # Adds a grid to the figure

    # Creating lines to animate in figure:
    n_lines = len(y)  # Number of lines to be plotted in animation
    lines = list()  # List of lines
    for num in range(n_lines):
        plot_kwargs = dict()  # Initialising dictionary of kwargs for plot function
        if use_legend:
            if num < num_legend:
                plot_kwargs['label'] = legend[num]  # If using legend
        if use_color:
            plot_kwargs['color'] = line_color[num]  # If changing line colors
        if use_style:
            plot_kwargs['linestyle'] = line_style[num]  # If changing line styles
        # Creating line:
        line, = ax.plot([], [], **plot_kwargs)
        lines.append(line)

    # Adding additional line if adding to plot animation:
    if x_add is not None:
        plot_kwargs = dict()  # Initialising dictionary of kwargs for plot function
        if use_legend:
            plot_kwargs['label'] = legend[-1]  # If using legend
        if use_color:
            plot_kwargs['color'] = line_color[-1]  # If changing line colors
        if use_style:
            plot_kwargs['linestyle'] = line_style[-1]  # If changing line styles
        line, = ax.plot([], [], **plot_kwargs)
        lines.append(line)

    # Adds legend if provided:
    if use_legend:
        ax.legend(fontsize=12, loc=legend_position)

    # Creating text to animate in figure:
    animation_text = ax.text(timepos[0], timepos[1], '', fontsize=timetextfontsize, transform=ax.transAxes)

    # Total number of frames of animation:
    n_frames = np.shape(y[0])[1]

    # Function to determine text to be animated:
    def text_to_animate(i, time, timetext):
        if time is not None:  # If there is time[i] to be animated
            time_str = '{:.2f}'.format(round(time[i], 2))  # Round time to 2 decimal places.
            if timetext is not None:  # If there is text to be added to time:
                output = timetext[0] + time_str + timetext[1]
            else:
                output = time_str
            return output

    # Initialising lines and text to animate:
    def init():
        for num in range(n_lines):  # Iterate through lines
            lines[num].set_data([], [])  # Initialising (x, y) values for each line
        if x_add is not None:  # Initialising additional line if adding to plot animation
            lines[n_lines].set_data([], [])  # Initialising (x_add, y_add) values
        animation_text.set_text('')  # Initialising text
        lines.append(animation_text)  # Attach animated text to list for output
        return lines

    # Animation function to plot i-th frame:
    def animate(i):
        for num in range(n_lines):  # Iterate through lines
            lines[num].set_data(x, y[num][:, i])  # Setting (x, y) values for each line
        if x_add is not None:  # Plotting additional line f adding to plot animation
            lines[n_lines].set_data(x_add, Y_add[:, i])  # Setting (x_add, y_add) values
        lines[-1].set_text(text_to_animate(i, time, timetext))  # Updating text (attached to lines list)
        return lines

    # Animating figure:
    ani = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=delay, blit=True)

    if save_ani:  # Saving animation:
        print('Saving animation...')
        Writer = writers['ffmpeg']
        writer = Writer(fps=save_fps)
        ani.save(save_name + '.mp4', writer=writer)

    if doing_mainloop:  # Automatically does mainloop() to plot GUI (turn this off if plotting multiple GUI / figures)
        print('Plotting animation...')
        tk.mainloop()


#######################################################
# Plot of 2D array as a colour plot / image:
def image_plot(x, y, n, **kwargs):

    #######################
    # Universal parameters:
    figsize = get_kwarg_value(kwargs, 'figsize', (6.00, 5.00))  # Size of figure
    xlabel = get_kwarg_value(kwargs, 'xlabel', '')  # x-axis label
    ylabel = get_kwarg_value(kwargs, 'ylabel', '')  # y-axis label
    xlimits = get_kwarg_value(kwargs, 'xlimits', [np.min(x), np.max(x)])  # Plot boundary limits for x-axis
    ylimits = get_kwarg_value(kwargs, 'ylimits', [np.min(y), np.max(y)])  # Plot boundary limits for y-axis
    xfontsize = get_kwarg_value(kwargs, 'xfontsize', 14)  # x-axis label font size
    yfontsize = get_kwarg_value(kwargs, 'yfontsize', 14)  # y-axis label font size
    ylabelrotation = get_kwarg_value(kwargs, 'ylabelrotation', 0)  # y-axis label rotation from axis
    ylabelpad = get_kwarg_value(kwargs, 'ylabelpad', 10)  # y-axis label spacing from axis
    ylabelcoords = get_kwarg_value(kwargs, 'ylabelcoords', None)  # y-axis label coordinates
    title = get_kwarg_value(kwargs, 'title', '')  # Title for plot
    titlefontsize = get_kwarg_value(kwargs, 'titlefontsize', 14)  # Title font size
    xscale = get_kwarg_value(kwargs, 'xscale', 'linear')  # Change scale of x-axis (linear or log)
    yscale = get_kwarg_value(kwargs, 'yscale', 'linear')  # Change scale of y-axis (linear or log)
    xticks = get_kwarg_value(kwargs, 'xticks', None)  # Plot x-tick labels
    yticks = get_kwarg_value(kwargs, 'yticks', None)  # Plot y-tick labels
    style = get_kwarg_value(kwargs, 'style', 'default')  # Style of figure
    image_min = get_kwarg_value(kwargs, 'image_min', None)  # Minimum of image colour
    image_max = get_kwarg_value(kwargs, 'image_max', None)  # Maximum of image colour
    cmap = get_kwarg_value(kwargs, 'cmap', None)  # Colour map of image
    cbarlabel = get_kwarg_value(kwargs, 'cbarlabel', '')  # Label of colorbar
    cbarfontsize = get_kwarg_value(kwargs, 'cbarfontsize', 12)  # Fontsize of colorbar
    cbarticks = get_kwarg_value(kwargs, 'cbarticks', None)  # Ticks of colorbar
    norm = get_kwarg_value(kwargs, 'norm', LogNorm())  # Scale of colorbar

    # Sets plot style:
    plt.style.use(style)

    # Creating figure and subplot axis:
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Clip range of n for vmin and vmax:
    n = n.clip(image_min, image_max)

    # Creating image:
    im = plt.pcolor(x, y, n, cmap=cmap, vmin=image_min, vmax=image_max, norm=norm)

    # Creating colour bar:
    if cbarticks is None:
        cbar = fig.colorbar(im, orientation='horizontal')
    else:
        cbar = fig.colorbar(im, ticks=cbarticks, orientation='horizontal')
        tick_labels = [str(tick) for tick in cbarticks]
        cbar.ax.set_xticklabels(tick_labels)
    cbar.set_label(cbarlabel, fontsize=cbarfontsize)

    # # Setting axis variables:
    ax.set_xlabel(xlabel, fontsize=xfontsize)  # Adds xlabel
    ax.set_ylabel(ylabel, fontsize=yfontsize, rotation=ylabelrotation, labelpad=ylabelpad)  # Adds ylabel
    if ylabelcoords is not None:
        ax.yaxis.set_label_coords(ylabelcoords[0], ylabelcoords[1])
    ax.set_title(title, fontsize=titlefontsize)  # Adds title
    ax.set_xlabel(xlabel)  # Adds xlabel
    ax.set_ylabel(ylabel)  # Adds ylabel
    ax.set_xlim(xlimits)  # Adds xlimits
    ax.set_ylim(ylimits)  # Adds ylimits
    ax.set_xscale(xscale)  # Changes x-axis scaling
    ax.set_yscale(yscale)  # Changes y-axis scaling
    if xticks is not None:
        plt.setp(ax, xticks=xticks, xticklabels=xticks)  # Modifies x-tick labels
    if yticks is not None:
        plt.setp(ax, yticks=yticks, yticklabels=yticks)  # Modifies y-tick labels

    # Show figure:
    plt.show()
