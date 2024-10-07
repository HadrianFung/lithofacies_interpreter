import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, LogFormatter
import ipywidgets as widgets
from ipywidgets import interact
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import display, HTML
import numpy as np
import pandas as pd
from PIL import Image
import os



def facies_plot(logs):
    """
    Function to plot the logs and facies

    Parameters
    ----------
    logs : pandas dataframe
        Dataframe containing the logs and facies data
    
    Returns
    -------
    None
    """
    # Create an HTML label for the dropdown
    html_label = widgets.HTML(value="<b style='font-size: 14px; '>Well Name:</b>")

    # Define your font style
    font = {
            'color':  'black',
            'size': 14,
        }
    
    
    well_names = logs.well.unique()
    
    name_grouped =logs.groupby('well',group_keys=True).apply(lambda x:x)

    # Create a dropdown widget for well name
    name_dropdown = widgets.Dropdown(options=well_names)
    well_name = name_dropdown.value

    def plot_by_well_name(well_name):
        # print(f'Well Name: {well_name}')
        
        if well_name not in well_names:
            display(HTML("<p style='font-size: 13px; color: black;'>Well name not found</p>"))
        
        else:
            logs = name_grouped.loc[well_name].copy()
            
            # Find depth range of non-NaN facies
            non_nan_facies_depths = logs[logs['facies'] != 'nan']['DEPTH']
            min_depth = non_nan_facies_depths.min()
            max_depth = non_nan_facies_depths.max()
            
            
            # Assuming facies_labels are defined globally or passed to the function
            facies_labels = ['os', 's', 'sh','ms']
            facies_colors = ['#F4D03F', '#85C1E9', '#A3E4D7', '#F5B7B1']  
            facies_num = [num for num, (labels, colors) in enumerate(zip(facies_labels, facies_colors))]
            
            # Create a new column for numerical facies
            logs['Facies_num'] = logs['facies'].apply(lambda x: facies_num[facies_labels.index(x)] if x in facies_labels else -1)
            
            # Sort the data by depth
            logs = logs.sort_values(by='DEPTH')
            cmap_facies = ListedColormap(facies_colors[:len(facies_labels)], name='facies_colormap')

            # Set the depth limits for the plot
            ztop=logs.DEPTH.min(); zbot=logs.DEPTH.max()
            #logs.RDEP = np.log10(logs.RDEP)
            #logs.RMED = np.log10(logs.RMED)
            #logs.RSHAL = np.log10(logs.RSHAL)
            #logs.permeability = np.log10(logs.permeability)
    
            # assing the axis for each track
            cluster = np.repeat(np.expand_dims(logs.loc[(logs.DEPTH >= min_depth) & (logs.DEPTH <= max_depth), 'Facies_num'].values, 1), 100, 1)
            fig = plt.figure(figsize=(30, 10))
            gs1 = GridSpec(1, 11, left=0.05, right=0.95, hspace=0.05, wspace=0.12,
                        width_ratios=[1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 0.6, 0.6])

            ax1 = fig.add_subplot(gs1[0, 1])
            ax2 = fig.add_subplot(gs1[0, 2])
            ax3 = fig.add_subplot(gs1[0, 3])
            ax4 = ax3.twiny()
            ax5 = ax3.twiny()
            ax6 = fig.add_subplot(gs1[0, 4])
            ax7 = ax6.twiny()
            ax8 = fig.add_subplot(gs1[0, 5])
            ax9 = ax8.twiny()
            ax10 = fig.add_subplot(gs1[0, 6])
            ax11 = fig.add_subplot(gs1[0, 7])
            ax12 = fig.add_subplot(gs1[0, 8])
            ax13 = fig.add_subplot(gs1[0, 9])
            ax14 = fig.add_subplot(gs1[0, 10])

            ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10,ax11,ax12,ax13,ax14]
            track_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11,ax12,ax13,ax14]
            
            
            for i, ax in enumerate(ax_list):
                if i > 0:  # This skips the first subplot which is assumed to be ax1
                    ax.set_yticklabels([])
                else:
                    # Only for the first subplot, set a larger font size for y-ticks
                    ax.tick_params(axis='y', labelsize=20)  # Set the font size for y-ticks
            
            
            desired_font_size = 12  # Replace 12 with whatever font size you prefer
            
            for ax in ax_list:
                plt.setp(ax.get_yticklabels(), visible=True)
                ax.set_ylim(max_depth, min_depth)  # Reverse order for depth axis
                
                # This will set the font size for the x-axis tick labels
                ax.tick_params(axis='x', labelsize=desired_font_size)
                
                # If you want to also set the font size for the x-axis label/title, do it here
                ax_title = ax.get_xlabel()  # Get the current label text
                ax.set_xlabel(ax_title, **font)
                
            
            #******************************************************************************************************
            ax1.plot(logs.GR, logs.DEPTH, '-r', alpha=.8, lw=0.5)
            ax1.set_xlabel("GR (API)")
            ax1.set_ylabel('Depth (ft)', **font)
            ax1.xaxis.set_label_position("top")
            ax1.xaxis.label.set_color("r")
            ax1.tick_params(axis='x', colors="r")
            ax1.spines["top"].set_edgecolor("r")
            ax1.spines["top"].set_position(("axes", 1.0))
            ax1.title.set_color('r')
            # *****************************************************************************************************
            ax2.plot(logs.PEF, logs.DEPTH, '-m', alpha=.8, lw=0.5)
            ax2.set_xlabel("PEF (b/e)")
            ax2.xaxis.set_label_position("top")
            ax2.xaxis.label.set_color("m")
            ax2.tick_params(axis='x', colors="m")
            ax2.spines["top"].set_edgecolor("m")
            ax2.spines["top"].set_position(("axes", 1.0))
            ax2.title.set_color('m')
            #******************************************************************************************************
            
            # Set the locator for major ticks only
            major_locator = LogLocator(base=10.0)
            # Set the formatter for tick labels
            formatter = LogFormatter(labelOnlyBase=True)  # True to format only base 10 numbers
            
            ax3.plot(logs.RDEP, logs.DEPTH, '-g', alpha=.8, lw=0.5)
            ax3.set_xlabel("RDEP (ohm.m)")
            ax3.set_xscale("log")
            ax3.xaxis.set_label_position("top")
            ax3.xaxis.label.set_color("k")
            ax3.xaxis.set_major_locator(major_locator)
            ax3.xaxis.set_major_formatter(formatter)
            ax3.grid(True, which='major', axis='x', color='0.8', linestyle='--', linewidth=0.5)
            ax3.set_xlim([1e-2, 1e2])  # Example range for RDEP, set according to your data
            ax3.tick_params(axis='x', colors="g")
            ax3.spines["top"].set_edgecolor("g")
            ax3.spines["top"].set_position(("axes", 1.0))
            ax3.title.set_color('g')
            #******************************************************************************************************
            ax4.plot(logs.RMED, logs.DEPTH, '-y', alpha=.8, lw=0.5)
            ax4.set_xlabel("RMED (ohm.m)")
            ax4.set_xscale("log")
            ax4.xaxis.label.set_color("y")
            ax4.xaxis.set_major_locator(major_locator)
            ax4.xaxis.set_major_formatter(formatter)
            ax4.grid(True, which='major', axis='x', color='0.8', linestyle='--', linewidth=0.5)
            ax4.set_xlim([1e-2, 1e2])
            ax4.tick_params(axis='x', colors="y")
            ax4.spines["top"].set_edgecolor("y")
            ax4.spines["top"].set_position(("axes", 1.08))
            ax4.title.set_color('y')
            #******************************************************************************************************
            ax5.plot(logs.RSHAL, logs.DEPTH, '-k', alpha=.8, lw=0.5)
            ax5.set_xlabel("RSHAL (ohm.m)")
            ax5.set_xscale("log")
            ax5.xaxis.label.set_color("k")
            ax5.xaxis.set_major_locator(major_locator)
            ax5.xaxis.set_major_formatter(formatter)
            ax5.grid(True, which='major', axis='x', color='0.8', linestyle='--', linewidth=0.5)
            ax5.set_xlim([1e-2, 1e2])
            ax5.tick_params(axis='x', colors="k")
            ax5.spines["top"].set_edgecolor("k")
            ax5.spines["top"].set_position(("axes", 1.16))
            ax5.title.set_color('k')
            
            for ax in [ax3, ax4, ax5]:
                ax.xaxis.label.set_size(14)

            #******************************************************************************************************
            ax6.plot(logs.NEUT, logs.DEPTH, '-b', alpha=.8, lw=0.5)
            ax6.set_xlabel("NPHI (v/v)")
            ax6.xaxis.label.set_color("b")
            ax6.tick_params(axis='x', colors="b")
            ax6.spines["top"].set_edgecolor("b")
            ax6.spines["top"].set_position(("axes", 1.08))
            ax6.title.set_color('b')
            ax6.set_xlim(0.45, -0.15)
            #******************************************************************************************************
            ax7.plot(logs.DENS, logs.DEPTH, '-c', alpha=.8, lw=0.5)
            ax7.set_xlabel("RHOB (g/cc)")
            ax7.xaxis.label.set_color("c")
            ax7.tick_params(axis='x', colors="c")
            ax7.spines["top"].set_edgecolor("c")
            ax7.spines["top"].set_position(("axes", 1.0))
            ax7.title.set_color('c')
            ax7.set_xlim(1.95, 2.95)
            ax7.set_xticks([1.65, 2,  2.4,   2.8])
            #******************************************************************************************************
            ax8.plot(logs.DTC, logs.DEPTH, '-k', alpha=.8, lw=0.5)
            ax8.set_xlabel("DTC (us/ft)")
            ax8.xaxis.label.set_color("k")
            ax8.tick_params(axis='x', colors="k")
            ax8.spines["top"].set_edgecolor("k")
            ax8.spines["top"].set_position(("axes", 1.08))
            ax8.title.set_color('k')
            ax8.set_xlim(400, 10)
            ax8.invert_yaxis()
            #******************************************************************************************************
            ax9.plot(logs.DTS, logs.DEPTH, '-g', alpha=.8, lw=0.5)
            ax9.set_xlabel("DTS (us/ft)")
            ax9.xaxis.label.set_color("g")
            ax9.tick_params(axis='x', colors="g")
            ax9.spines["top"].set_edgecolor("g")
            ax9.spines["top"].set_position(("axes", 1.0))
            ax9.title.set_color('g')
            ax9.set_xlim(400, 10)
            ax9.invert_yaxis()
            #******************************************************************************************************
            ax10.plot(logs.SP, logs.DEPTH, '-b', alpha=.8, lw=0.5)
            ax10.set_xlabel("SP (mV)")
            ax10.xaxis.set_label_position("top")
            ax10.xaxis.label.set_color("b")
            ax10.tick_params(axis='x', colors="b")
            ax10.spines["top"].set_edgecolor("b")
            ax10.spines["top"].set_position(("axes", 1.0))
            ax10.title.set_color('b')
            #******************************************************************************************************
            ax11.plot(logs.porosity, logs.DEPTH, '-g', alpha=.8, lw=0.5)
            ax11.set_xlabel("PHI (%)")
            ax11.xaxis.label.set_color("g")
            ax11.tick_params(axis='x', colors="g")
            ax11.spines["top"].set_edgecolor("g")
            ax11.spines["top"].set_position(("axes", 1.0))
            ax11.title.set_color('g')
            ax11.set_xlim(0, 100)
            #******************************************************************************************************
            ax12.plot(logs.permeability, logs.DEPTH, '-g', alpha=.8, lw=0.5)
            ax12.set_xlabel("PERM (mD)")
            ax12.xaxis.label.set_color("g")
            ax12.tick_params(axis='x', colors="g")
            ax12.spines["top"].set_edgecolor("g")
            ax12.spines["top"].set_position(("axes", 1.0))
            ax12.title.set_color('g')
            ax12.set_xscale("log")
            ax12.set_xlim(1, 10000)
            #******************************************************************************************************
            
            # Plot the facies log              
            im = ax13.imshow(cluster, interpolation='none', aspect='auto',
            cmap=cmap_facies, vmin=0, vmax=len(facies_labels)-1,
            extent=[0, 1, max_depth, min_depth])  # Set the extent for the depth range                
            
            ax13.set_xlabel('facies', fontsize=desired_font_size)
            ax13.set_yticklabels([])
            ax13.set_xticks([])
            
            # Add a colorbar
            divider = make_axes_locatable(ax13)
            cax = divider.append_axes("right", size="20%", pad=0.05)
            cbar=plt.colorbar(im, cax=cax)
            
            # Set the tick marks properly
            num_labels = len(facies_labels)
            tick_locs = (np.arange(num_labels) + 0.5)*(num_labels-1)/num_labels
            cbar.set_ticks(tick_locs)
            
            # Set custom tick labels
            cbar.set_ticklabels(facies_labels)
            cbar.ax.set_yticklabels(facies_labels, **font, rotation=0)
            
            # Shift labels to the left
            x_offset = 0.6  # Adjust this value as needed
            for label in cbar.ax.yaxis.get_ticklabels():
                label.set_horizontalalignment('left')
                label.set_position((x_offset, 0))
            
            
            # Set the label size for the colorbar
            for ax in ax_list:
                ax.xaxis.set_ticks_position("top")
                ax.xaxis.set_label_position("top")               

            for ax in track_list:
                ax.grid(True, color='0.8', dashes=(5,2,1,2))
                ax.set_facecolor('#ffffed')
                                
            image_file_path = logs.loc[logs['well'] == well_name, 'image_file'].iloc[0]
            depths_file_path = logs.loc[logs['well'] == well_name, 'depth_file'].iloc[0]
            
            # The paths in the dataframe are relative, we will convert them to absolute paths
            # If your script is running from a different directory, you might need to adjust the paths
            image_file_path = os.path.join(image_file_path)
            depths_file_path = os.path.join(depths_file_path)
            
            # Load the image and depths data
            depths = np.load(depths_file_path)
            image_array = np.load(image_file_path)
            
            # Assuming the subplot ax14 is already defined in your code
            top_depth = depths[0]
            bottom_depth = depths[-1]
            
            # Display the image with the depth range matching the depths of the well log
            ax14.imshow(image_array, aspect='auto', extent=[0, image_array.shape[1], bottom_depth, top_depth])

            # Invert the y-axis to match the depth increasing with depth
            ax14.invert_yaxis()

            # Hide x and y ticks for the core image subplot
            ax14.set_xticks([])
            ax14.set_yticks([])

            # Add a label for the core image subplot if desired
            ax14.set_xlabel('Core Image')
            
            plt.tight_layout()

    interact(plot_by_well_name, well_name=name_dropdown)

